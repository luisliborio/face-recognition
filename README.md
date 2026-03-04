Projeto de reconhecimento facial com few-shot learning e softmargin triplet loss para Aula 5 de Aplicações em DS.

# Opção 1: Treinar seu próprio modelo
1. dataset CelebA
    - images/              # pasta com todas as imagens e nomes originais
    - identity_CelebA.txt  # arquivo de anotações com caminhos e IDs das imagens
2. clonar repositório git
    - git clone git@github.com:luisliborio/face-recognition.git
    - criar arquivo .env
        - POSTGRES_DB=facedb
        - POSTGRES_USER=admin
        - POSTGRES_PASSWORD=senha_super_segura
3. treinar modelo
    - treinar seu modelo em "notebooks/face-recognition-modeling.ipynb"
    - copiar artefato model.keras para backend-api/
4. subir aplicação docker no terminal (dentro da pasta do projeto!)
    - na pasta do projeto, face-recognition/
    - docker compose up -d --build
    - localhost:8080


# Opção 2: Puxar imagens do app via docker hub (_lsmliborio/_)

*As imagem do backend dockerhub já tem um modelo treinado built-in*

1. Crie uma pasta vazia (e.g., face-recognition/)
    - criar arquivo .env
        - POSTGRES_DB=facedb
        - POSTGRES_USER=admin
        - POSTGRES_PASSWORD=senha_super_segura
2. copiar docker-compose(dockerhub).yml (no .zip ou github)
    - renomear para docker-compose.yml
3. Executar o compose no terminal (dentro da pasta do projeto!)
    - docker compose up -d

---

# Instruções de entrega
1. Monte o app em sua máquina
    - docker compose up -d # *(de acordo com Opção 1 ou 2)*
    - agora você pode ver as imagens usando _docker images_
2. Adicione as tags do seu repositório DOCKER HUB
    - docker tag face-recognition-frontend:latest SEU-REPO-DOCKERHUB/facerec-frontend:latest
    - docker tag face-recognition-backend-api:latest SEU-REPO-DOCKERHUB/facerec-backend:latest
3. Faça o upload de suas imagens para o seu DOCKER HUB
    - docker login
    - docker push SEU-REPO-DOCKERHUB/facerec-frontend:latest
    - docker push SEU-REPO-DOCKERHUB/facerec-backend:latest


## Resumo de Comandos necessários:
- docker compose up -d --build   # subir aplicação pela primeira vez a partir do código fonte e docker-compose.yml
- docker compose up -d           # subir aplicação pela segunda vez em diante, ou diretamente a partir das imagens no docker hub
- docker compose down            # derrubar a aplicação
- docker compose down -v         # derrubar a aplicação e base de dados
- tag NOME-DA-IMAGEM-ATUAL:TAG_ATUAL NOME-DA-IMAGEM-NOVA:TAG_NOVA # modificar nome de uma imagem:tag
- docker login
- docker push SEU-DOCKERHUB/SUA-IMAGE:TAG # upload para seu repositório remoto