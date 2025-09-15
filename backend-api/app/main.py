from fastapi import FastAPI

app = FastAPI(title="API de Reconhecimento Facial")

@app.get("/")
def read_root():
    """Endpoint raiz para verificar se a API está no ar."""
    return {"status": "ok", "message": "API de Reconhecimento Facial está funcionando!"}
