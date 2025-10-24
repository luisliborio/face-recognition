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

# VERIFICATION_THRESHOLD: valor limiar para decidir se uma comparação é "match".
# Observação didática: esse valor depende MUITO do modelo, dos dados e de como a
# similaridade/distância é calculada. Em produção você deve calibrar isso com
# um conjunto de validação (avaliando FPR/FNR).
VERIFICATION_THRESHOLD = -0.55 # NOTE: calibração direta do coeficient J durante treino

# --- Inicialização da Aplicação FastAPI ---
# FastAPI cria automaticamente documentação interativa (Swagger) e gerencia rotas.
app = FastAPI(title="API de Reconhecimento Facial")

# --- Configuração do CORS ---
# CORS (Cross-Origin Resource Sharing) controla quais origens (domínios)
# podem fazer requisições para essa API. Para facilitar o desenvolvimento,
# allow_origins="*" permite qualquer origem — isso é inseguro em produção.
# Recomendação: Em produção, trocar pelo domínio do frontend
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Carregamento do Modelo ---
# O modelo é carregado na memória uma única vez quando a API inicia.
# Isso evita carregar o modelo a cada requisição, o que seria muito lento.
# A função load_keras_model está em model_loader.py — ela deve retornar
# um objeto com o atributo `feature_extractor` que faz .predict().
model = load_keras_model(os.getenv("MODEL_PATH", "/app/model.keras"))

# --- Modelos de Dados (Pydantic) ---
# Pydantic valida automaticamente os dados de entrada/saída das rotas.
# Isso ajuda a detectar erros cedo e a documentar os tipos esperados.

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
    # Em ambientes distribuídos (Docker + DB separado) às vezes o DB ainda
    # não está pronto quando a API sobe. Por isso há retries.
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
            # Em caso de falha, aguarda 2 segundos e tenta novamente.
            print(f"Erro ao conectar ao DB: {e}. Tentando novamente em 2 segundos...")
            retries -= 1
            time.sleep(2)
    # Se esgotarem as tentativas, levantamos uma exceção clara.
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
            # Isso gera colunas embedding_0, embedding_1, ..., embedding_{EMBEDDING_DIM-1}
            # Observação didática: outra opção é armazenar o embedding como um array
            # (ex.: tipo JSON/ARRAY) dependendo do banco; Escolhi separar em colunas 
            # tradicionais para facilitar o query SQL simples.
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
            # preprocess_image: deve decodificar base64 -> PIL.Image -> np.array normalizado
            # Retorna um np.ndarray com shape (H, W, 3).
            processed_img = preprocess_image(base64_img) # np (H, W, 3)
            # aumenta cada imagem 5 vezes
            # flip, crop, brilho, etc, em baixa intensidade.            
            for _ in range(5):                
                input_images.append(augment(processed_img))

        except Exception as e:
            # Se alguma imagem falhar ao ser processada, retornamos 400 ao cliente.
            raise HTTPException(status_code=400, detail=f"Erro ao processar uma das imagens: {e}")
    
    # Agrupamos todas as imagens em um tensor (N, H, W, 3)
    # Observação importante: todas as imagens devem ter o mesmo tamanho após preprocessamento.
    input_tensor = np.stack(input_images, axis=0).astype('float32') # (N, H, W, 3)
    print(f'Input tensor gerado. shape: {input_tensor.shape}')
    
    # inferência e avg support set embedding
    try: 
        # espera um array (N, H, W, 3) e retorna (N, EMBEDDING_DIM)
        embeddings = model.feature_extractor.predict(input_tensor)
        # Calcula a média dos embeddings para criar um "centro" representativo do usuário.
        mean_embedding = embeddings.mean(axis=0)
        print(f'Support Embedding carregado. Shape: {mean_embedding.shape}')

    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Erro ao realizar inferência: {e}')

    # Salva no banco de dados
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Monta os nomes das colunas embedding_0 ... embedding_{D-1}
            embedding_cols_names = ", ".join([f"embedding_{i}" for i in range(EMBEDDING_DIM)])
            # Placeholders para o psycopg2 (%s) — atenção à ordem dos parâmetros ao executar.
            embedding_values_placeholders = ", ".join(["%s"] * EMBEDDING_DIM)
            
            insert_query = f"""
            INSERT INTO users (name, {embedding_cols_names})
            VALUES (%s, {embedding_values_placeholders})
            RETURNING id;
            """
            
            # Passamos primeiro o nome, depois cada valor do embedding (unpacked).
            # mean_embedding.tolist() converte o np.array em lista de floats.
            cur.execute(insert_query, (request.name, *mean_embedding.tolist()))
            user_id = cur.fetchone()['id']
            conn.commit()
        return {"status": "success", "user_id": user_id}
    except Exception as e:
        # Erros de banco resultam em 500 (erro no servidor).
        raise HTTPException(status_code=500, detail=f"Erro no banco de dados: {e}")
    finally:
        if conn:
            conn.close()


@app.post("/verify", response_model=VerifyResponse)
def verify_user(request: VerifyRequest):
    """Verifica uma imagem contra todos os usuários cadastrados."""
    # Processa a imagem de teste
    try:
        # Passa a string base64 diretamente para a função de pré-processamento.
        processed_img = preprocess_image(request.image)
        # Para robustez, aplica 20 augmentações na imagem de consulta e usa a primeira embedding.
        input_tensor = np.stack([augment(processed_img) for _ in range(20)], axis=0).astype('float32')
        query_embedding = model.feature_extractor.predict(input_tensor).mean(axis=0)        
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
        # Se não há usuários cadastrados, não há como identificar.
        return {"identified": False}

    # Calcula as distâncias
    # Constrói uma matriz (num_users, EMBEDDING_DIM) a partir das colunas do DB.
    db_embeddings = np.array([[user[f'embedding_{i}'] for i in range(EMBEDDING_DIM)] for user in users])
    # Aqui a "distância" usada é -sum((db - query)^2)
    # Ou seja, é o negativo da soma dos quadrados (mais próximo => valor maior, menos negativo).
    distances = -np.sum((db_embeddings - query_embedding) ** 2, axis=1)
    
    # Seleciona o usuário com maior "similaridade" (maior distância negada).
    max_distance_idx = np.argmax(distances)
    max_distance = distances[max_distance_idx]
    
    # Compara com o limiar para decidir identificação.
    if max_distance >= VERIFICATION_THRESHOLD:
        identified_user = users[max_distance_idx]
        return {
            "identified": True,
            "name": identified_user['name'],
            "distance": float(max_distance)
        }
    else:
        # Retorna identificado False e a maior distância encontrada para ajudar debug/calibração.
        return {
            "identified": False,
            "distance": float(max_distance)
        }
