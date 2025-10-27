Projeto de reconhecimento facial com few-shot learning e softmargin triplet loss para Aula 5 de Aplicações em DS.

# Imagens docker hub:
- backend:  lsmliborio/facerec-backend
- frontend: lsmliborio/facerec-frontend

## caso rode o sistema a partir do dockerhub, não esqueça de alterar o docker-compose.yml com as respectivas imagens

## As imagem do backend já tem um modelo treinado built-in


# Comandos:
- docker compose up -d --build // subir aplicação pela primeira vez a partir do código fonte e docker-compose.yml
- docker compose up -d         // subir aplicação pela segunda vez em diante, ou diretamente a partir das imagens no docker hub
- docker compose down          // derrubar a aplicação
- docker compose down -v       // derrubar a aplicação e base de dados
