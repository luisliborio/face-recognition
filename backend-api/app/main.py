import os
import time
from typing import List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np

from .model_loader import load_keras_model, preprocess_image, augment

# --- Configurações e Constantes ---
EMBEDDING_DIM = 512
VERIFICATION_THRESHOLD = -0.00075 # TODO: faltando calibração

# --- Inicialização da Aplicação FastAPI ---
app = FastAPI(title="API de Reconhecimento Facial")

# --- Configuração do CORS ---
# Isso permite que o frontend (rodando em outra porta/origem) se comunique com a API.
origins = ["*"]  # Em produção, restrinja para o domínio do seu frontend

# ???
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Carregamento do Modelo ---
# O modelo é carregado na memória uma única vez quando a API inicia.
model = load_keras_model(os.getenv("MODEL_PATH", "/app/model.keras"))

# --- Modelos de Dados (Pydantic) ---
# Define a estrutura esperada para as requisições e respostas.

class RegisterRequest(BaseModel):
    name: str
    images: List[str]  # Lista de imagens em base64

class VerifyRequest(BaseModel):
    image: str # Imagem em base64

class RegisterResponse(BaseModel):
    status: str
    user_id: int

class VerifyResponse(BaseModel):
    identified: bool
    name: Optional[str] = None
    distance: Optional[float] = None

# --- Conexão com o Banco de Dados ---
def get_db_connection():
    """Tenta se conectar ao banco de dados com múltiplas tentativas."""
    retries = 5
    while retries > 0:
        try:
            print('estabelecendo conexão com base de dados...')
            conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "database"),
                dbname=os.getenv("POSTGRES_DB"),
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD"),
                cursor_factory=RealDictCursor
            )
            print("Conexão com o banco de dados estabelecida com sucesso.")
            return conn
        except psycopg2.OperationalError as e:
            print(f"Erro ao conectar ao DB: {e}. Tentando novamente em 5 segundos...")
            retries -= 1
            time.sleep(5)
    raise Exception("Não foi possível conectar ao banco de dados após várias tentativas.")

# --- Eventos de Inicialização ---
@app.on_event("startup")
def startup_event():
    """Executa tarefas na inicialização da API."""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Cria a tabela de usuários dinamicamente com base na dimensão do embedding
            embedding_cols = ", ".join([f"embedding_{i} FLOAT" for i in range(EMBEDDING_DIM)])
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                {embedding_cols}
            );
            """
            cur.execute(create_table_query)
            conn.commit()
            print("Tabela 'users' verificada/criada com sucesso.")
    finally:
        if conn:
            conn.close()

# --- Endpoints da API ---

@app.get("/")
def read_root():
    """Endpoint inicial para verificar se a API está funcionando."""
    return {"status": "ok", "message": "API de Reconhecimento Facial está funcionando!"}


@app.post("/register", response_model=RegisterResponse)
def register_user(request: RegisterRequest):
    """Cadastra um novo usuário a partir de múltiplas fotos."""
    # preprocessa todas as imagens em um único tensor de inputs
    input_images = []      
    for base64_img in request.images:
        try:            
            processed_img = preprocess_image(base64_img) # (H, W, 3)
            for _ in range(5):
                print(f'processed image dtype: {processed_img.dtype}')
                input_images.append(augment(processed_img))

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro ao processar uma das imagens: {e}")
    
    input_tensor = np.stack(input_images, axis=0).astype('float32') # (N, H, W, 3)
    print(f'Input tensor gerado. shape: {input_tensor.shape}')
    
    # inferência e avg support set embedding
    try: 
        embeddings = model.feature_extractor.predict(input_tensor)
        mean_embedding = embeddings.mean(axis=0)
        print(f'Support Embedding carregado. Shape: {mean_embedding.shape}')

    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Erro ao realizar inferência: {e}')

    # Salva no banco de dados
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            embedding_cols_names = ", ".join([f"embedding_{i}" for i in range(EMBEDDING_DIM)])
            embedding_values_placeholders = ", ".join(["%s"] * EMBEDDING_DIM)
            
            insert_query = f"""
            INSERT INTO users (name, {embedding_cols_names})
            VALUES (%s, {embedding_values_placeholders})
            RETURNING id;
            """
            
            cur.execute(insert_query, (request.name, *mean_embedding.tolist()))
            user_id = cur.fetchone()['id']
            conn.commit()
        return {"status": "success", "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no banco de dados: {e}")
    finally:
        if conn:
            conn.close()


@app.post("/verify", response_model=VerifyResponse)
def verify_user(request: VerifyRequest):
    """Verifica uma imagem contra todos os usuários cadastrados."""
    # Processa a imagem de teste
    try:
        # CORRIGIDO: Passa a string base64 diretamente para a função de pré-processamento.
        processed_img = preprocess_image(request.image)
        input_tensor = np.stack([augment(processed_img) for _ in range(5)], axis=0).astype('float32')
        query_embedding = model.feature_extractor.predict(input_tensor)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar imagem de verificação: {e}")

    # Busca todos os embeddings do banco de dados
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, " + ", ".join([f"embedding_{i}" for i in range(EMBEDDING_DIM)]) + " FROM users")
            users = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar usuários no DB: {e}")
    finally:
        if conn:
            conn.close()
    
    if not users:
        return {"identified": False}

    # Calcula as distâncias
    db_embeddings = np.array([[user[f'embedding_{i}'] for i in range(EMBEDDING_DIM)] for user in users])
    distances = -np.mean((db_embeddings - query_embedding) ** 2, axis=1)
    
    max_distance_idx = np.argmax(distances)
    max_distance = distances[max_distance_idx]
    
    if max_distance >= VERIFICATION_THRESHOLD:
        identified_user = users[max_distance_idx]
        return {
            "identified": True,
            "name": identified_user['name'],
            "distance": float(max_distance)
        }
    else:
        return {
            "identified": False,
            "distance": float(max_distance)
        }
