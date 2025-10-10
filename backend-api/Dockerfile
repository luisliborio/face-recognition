# Usar uma imagem base do Python
FROM lsmliborio/tensorflow:2.17.0.2

# Definir o diretório de trabalho dentro do container
WORKDIR /app

# Copiar o arquivo de dependências e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante dos arquivos da aplicação
COPY . .

# Expor a porta que a API irá rodar
EXPOSE 8000

# Comando para iniciar a aplicação quando o container for executado
# O reload é útil para o ambiente de desenvolvimento
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
