// O código só começa a rodar quando todo o conteúdo HTML da página for carregado
document.addEventListener('DOMContentLoaded', () => {

    // ==============================
    // SEÇÃO 1: Referências aos Elementos da Interface (UI)
    // ==============================
    // Aqui pegamos os elementos da página (HTML) que serão manipulados com JavaScript

    const homeScreen = document.getElementById('home-screen');   // Tela inicial
    const mainApp = document.getElementById('main-app');         // Tela principal (após iniciar)
    const webcamVideo = document.getElementById('webcam');       // Vídeo da webcam
    const overlayCanvas = document.getElementById('overlay');    // Camada de desenho (borda verde/vermelha)
    const captureCanvas = document.getElementById('capture-canvas'); // Canvas onde a foto é capturada
    const nameInput = document.getElementById('name-input');     // Campo de texto para nome
    const message = document.getElementById('message');          // Mensagens de status (ex: sucesso, erro)
    const photoGallery = document.getElementById('photo-gallery'); // Mostra as fotos capturadas
    const appTitle = document.getElementById('app-title');       // Título que muda conforme o modo

    // ==============================
    // SEÇÃO 2: Referências aos Botões
    // ==============================
    const startRegisterBtn = document.getElementById('start-register-btn'); // Botão “Cadastrar”
    const startVerifyBtn = document.getElementById('start-verify-btn');     // Botão “Verificar”
    const captureBtn = document.getElementById('capture-btn');              // Botão “Capturar foto”
    const submitBtn = document.getElementById('submit-btn');                // Botão “Enviar/Concluir”
    const backBtn = document.getElementById('back-btn');                    // Botão “Voltar”

    // ==============================
    // SEÇÃO 3: Estado da Aplicação
    // ==============================
    // Aqui guardamos informações que mudam durante o uso da aplicação
    let currentMode = null;   // Indica o modo atual: 'register' (cadastrar) ou 'verify' (verificar)
    let capturedPhotos = [];  // Guarda as fotos capturadas da webcam
    let stream = null;        // Armazena o fluxo de vídeo da webcam

    const API_URL = 'http://localhost:8000'; // Endereço onde o backend (FastAPI) está rodando

    // ==============================
    // SEÇÃO 4: Inicialização e Controle da Interface
    // ==============================

    // Função que configura todos os botões da tela
    const init = () => {
        // Quando o botão “Cadastrar” é clicado, inicia o modo de cadastro
        startRegisterBtn.addEventListener('click', () => startApp('register'));
        // Quando o botão “Verificar” é clicado, inicia o modo de verificação
        startVerifyBtn.addEventListener('click', () => startApp('verify'));
        // Botão “Voltar” reinicia a aplicação
        backBtn.addEventListener('click', resetApp);
        // Botão “Capturar foto” tira uma foto da webcam
        captureBtn.addEventListener('click', capturePhoto);
        // Botão “Enviar” envia os dados para o servidor
        submitBtn.addEventListener('click', submitData);
    };

    // Função que inicia o aplicativo no modo escolhido
    const startApp = (mode) => {
        currentMode = mode; // guarda o modo atual
        homeScreen.classList.add('hidden'); // esconde a tela inicial
        mainApp.classList.remove('hidden'); // mostra a tela principal
        updateUIForMode();  // atualiza o visual conforme o modo
        startWebcam();      // liga a webcam
    };

    // Altera o texto e os botões conforme o modo (cadastro ou verificação)
    const updateUIForMode = () => {
        if (currentMode === 'register') {
            appTitle.textContent = 'Modo de Cadastro';
            nameInput.classList.remove('hidden');
            submitBtn.classList.remove('hidden');
            submitBtn.textContent = 'Concluir Cadastro';
            captureBtn.textContent = 'Capturar Foto (0)';
        } else { // Modo de verificação
            appTitle.textContent = 'Modo de Verificação';
            nameInput.classList.add('hidden');
            submitBtn.classList.add('hidden');
            captureBtn.textContent = 'Verificar Rosto';
        }
    };

    // Reseta toda a aplicação e volta para a tela inicial
    const resetApp = () => {
        stopWebcam();                  // desliga a webcam
        currentMode = null;            // limpa o modo atual
        capturedPhotos = [];           // apaga as fotos
        photoGallery.innerHTML = '';   // limpa a galeria
        message.textContent = '';      // limpa mensagens
        nameInput.value = '';          // apaga o nome digitado
        clearOverlay();                // limpa o canvas de destaque
        mainApp.classList.add('hidden');
        homeScreen.classList.remove('hidden');
    };

    // ==============================
    // SEÇÃO 5: Webcam e Canvas
    // ==============================

    // Liga a webcam e mostra o vídeo na tela
    const startWebcam = async () => {
        try {
            // Pede permissão e acessa a webcam
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            webcamVideo.srcObject = stream;

            // Quando o vídeo carrega, ajusta os tamanhos dos canvas
            webcamVideo.onloadedmetadata = () => {
                overlayCanvas.width = webcamVideo.videoWidth;
                overlayCanvas.height = webcamVideo.videoHeight;
                captureCanvas.width = webcamVideo.videoWidth;
                captureCanvas.height = webcamVideo.videoHeight;
            };
        } catch (err) {
            console.error("Erro ao acessar a webcam: ", err);
            message.textContent = 'Não foi possível acessar a webcam.';
            message.style.color = '#ff4d4d';
        }
    };

    // Desliga a webcam
    const stopWebcam = () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    };

    // Captura uma foto do vídeo da webcam
    const capturePhoto = () => {
        const context = captureCanvas.getContext('2d');
        // Desenha no canvas o frame atual do vídeo
        context.drawImage(webcamVideo, 0, 0, captureCanvas.width, captureCanvas.height);
        
        // Converte o conteúdo do canvas em uma imagem base64
        const base64ImageFull = captureCanvas.toDataURL('image/jpeg'); // inclui cabeçalho
        const base64ImageOnly = base64ImageFull.split(',')[1]; // apenas os dados da imagem

        if (currentMode === 'register') {
            // Guarda a foto e mostra na galeria
            capturedPhotos.push(base64ImageOnly);
            addPhotoToGallery(base64ImageFull);
            captureBtn.textContent = `Capturar Foto (${capturedPhotos.length})`;
        } else { // Modo verificação
            capturedPhotos = [base64ImageOnly]; // usa apenas uma foto
            submitData(); // envia para o backend para verificar
        }
    };

    // Adiciona a foto capturada na galeria de miniaturas
    const addPhotoToGallery = (imageDataUrl) => {
        const img = document.createElement('img');
        img.src = imageDataUrl;
        img.classList.add('thumbnail');
        photoGallery.appendChild(img);
    };

    // Desenha uma borda verde (sucesso) ou vermelha (erro) sobre o vídeo
    const drawResultBox = (isSuccess) => {
        const ctx = overlayCanvas.getContext('2d');
        clearOverlay();
        ctx.strokeStyle = isSuccess ? '#00e676' : '#ff4d4d';
        ctx.lineWidth = 5;
        ctx.strokeRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    };

    // Limpa o canvas de sobreposição
    const clearOverlay = () => {
        const ctx = overlayCanvas.getContext('2d');
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    };

    // ==============================
    // SEÇÃO 6: Comunicação com a API Backend
    // ==============================

    // Envia os dados (fotos e nome) para o backend via HTTP
    const submitData = async () => {
        // Verificação básica antes de enviar
        if (currentMode === 'register' && (!nameInput.value || capturedPhotos.length === 0)) {
            message.textContent = 'Por favor, digite um nome e capture pelo menos uma foto.';
            message.style.color = '#ff4d4d';
            return;
        }

        let endpoint = ''; // caminho da API
        let payload = {};  // conteúdo a ser enviado

        // Define os dados conforme o modo
        if (currentMode === 'register') {
            endpoint = '/register';
            payload = { name: nameInput.value, images: capturedPhotos };
        } else { // verify
            endpoint = '/verify';
            payload = { image: capturedPhotos[0] };
        }

        message.textContent = 'Processando...';
        message.style.color = '#e0e0e0';

        try {
            // Envia os dados ao backend (requisição HTTP POST)
            const response = await fetch(`${API_URL}${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            // Converte a resposta em JSON
            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || 'Ocorreu um erro.');
            }

            // Trata o resultado retornado pela API
            handleApiResponse(result);

        } catch (error) {
            console.error('Erro na API:', error);
            message.textContent = `Erro: ${error.message}`;
            message.style.color = '#ff4d4d';
            if (currentMode === 'verify') drawResultBox(false);
        }
    };

    // Recebe e interpreta a resposta vinda da API
    const handleApiResponse = (result) => {
        if (currentMode === 'register') {
            message.textContent = `Usuário '${nameInput.value}' cadastrado com sucesso!`;
            message.style.color = '#00e676';
            setTimeout(resetApp, 3000); // volta à tela inicial após 3 segundos
        } else { // verify
            if (result.identified) {
                message.textContent = `Identificado: ${result.name} (Distância: ${result.distance.toFixed(6)})`;
                message.style.color = '#00e676';
                drawResultBox(true);
            } else {
                message.textContent = 'Usuário não identificado.';
                message.style.color = '#ff4d4d';
                drawResultBox(false);
            }
        }
    };

    // ==============================
    // SEÇÃO 7: Inicialização Final
    // ==============================
    // Chama a função que liga todos os botões e inicia o app
    init();
});
