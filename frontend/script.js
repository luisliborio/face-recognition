document.addEventListener('DOMContentLoaded', () => {
    // Elementos da UI
    const homeScreen = document.getElementById('home-screen');
    const mainApp = document.getElementById('main-app');
    const webcamVideo = document.getElementById('webcam');
    const overlayCanvas = document.getElementById('overlay');
    const captureCanvas = document.getElementById('capture-canvas');
    const nameInput = document.getElementById('name-input');
    const message = document.getElementById('message');
    const photoGallery = document.getElementById('photo-gallery');
    const appTitle = document.getElementById('app-title');

    // Botões
    const startRegisterBtn = document.getElementById('start-register-btn');
    const startVerifyBtn = document.getElementById('start-verify-btn');
    const captureBtn = document.getElementById('capture-btn');
    const submitBtn = document.getElementById('submit-btn');
    const backBtn = document.getElementById('back-btn');

    // Estado da aplicação
    let currentMode = null; // 'register' ou 'verify'
    let capturedPhotos = [];
    let stream = null;

    const API_URL = 'http://localhost:8000'; // URL da nossa API Backend

    // --- Funções de Inicialização e UI ---

    const init = () => {
        startRegisterBtn.addEventListener('click', () => startApp('register'));
        startVerifyBtn.addEventListener('click', () => startApp('verify'));
        backBtn.addEventListener('click', resetApp);
        captureBtn.addEventListener('click', capturePhoto);
        submitBtn.addEventListener('click', submitData);
    };

    const startApp = (mode) => {
        currentMode = mode;
        homeScreen.classList.add('hidden');
        mainApp.classList.remove('hidden');
        updateUIForMode();
        startWebcam();
    };

    const updateUIForMode = () => {
        if (currentMode === 'register') {
            appTitle.textContent = 'Modo de Cadastro';
            nameInput.classList.remove('hidden');
            submitBtn.classList.remove('hidden');
            submitBtn.textContent = 'Concluir Cadastro';
            captureBtn.textContent = 'Capturar Foto (0)';
        } else { // verify
            appTitle.textContent = 'Modo de Verificação';
            nameInput.classList.add('hidden');
            submitBtn.classList.add('hidden');
            captureBtn.textContent = 'Verificar Rosto';
        }
    };

    const resetApp = () => {
        stopWebcam();
        currentMode = null;
        capturedPhotos = [];
        photoGallery.innerHTML = '';
        message.textContent = '';
        nameInput.value = '';
        clearOverlay();
        
        mainApp.classList.add('hidden');
        homeScreen.classList.remove('hidden');
    };

    // --- Funções da Webcam e Canvas ---

    const startWebcam = async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            webcamVideo.srcObject = stream;
            webcamVideo.onloadedmetadata = () => {
                // Ajusta o tamanho do canvas para o tamanho do vídeo
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

    const stopWebcam = () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    };

    const capturePhoto = () => {
        const context = captureCanvas.getContext('2d');
        context.drawImage(webcamVideo, 0, 0, captureCanvas.width, captureCanvas.height);
        
        // Converte a imagem para base64
        const base64Image = captureCanvas.toDataURL('image/jpeg').split(',')[1];
        
        if (currentMode === 'register') {
            capturedPhotos.push(base64Image);
            addPhotoToGallery(captureCanvas.toDataURL('image/jpeg'));
            captureBtn.textContent = `Capturar Foto (${capturedPhotos.length})`;
        } else { // verify
            capturedPhotos = [base64Image]; // Apenas uma foto para verificação
            submitData(); // Envia para verificação imediatamente
        }
    };

    const addPhotoToGallery = (imageDataUrl) => {
        const img = document.createElement('img');
        img.src = imageDataUrl;
        img.classList.add('thumbnail');
        photoGallery.appendChild(img);
    };

    const drawResultBox = (isSuccess) => {
        const ctx = overlayCanvas.getContext('2d');
        clearOverlay();
        
        ctx.strokeStyle = isSuccess ? '#00e676' : '#ff4d4d'; // Verde ou Vermelho
        ctx.lineWidth = 5;
        ctx.strokeRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    };

    const clearOverlay = () => {
        const ctx = overlayCanvas.getContext('2d');
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    };

    // --- Funções de Comunicação com a API ---

    const submitData = async () => {
        if (currentMode === 'register' && (!nameInput.value || capturedPhotos.length === 0)) {
            message.textContent = 'Por favor, digite um nome e capture pelo menos uma foto.';
            message.style.color = '#ff4d4d';
            return;
        }

        let endpoint = '';
        let payload = {};

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
            const response = await fetch(`${API_URL}${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || 'Ocorreu um erro.');
            }

            handleApiResponse(result);

        } catch (error) {
            console.error('Erro na API:', error);
            message.textContent = `Erro: ${error.message}`;
            message.style.color = '#ff4d4d';
            if (currentMode === 'verify') drawResultBox(false);
        }
    };

    const handleApiResponse = (result) => {
        if (currentMode === 'register') {
            message.textContent = `Usuário '${nameInput.value}' cadastrado com sucesso!`;
            message.style.color = '#00e676';
            setTimeout(resetApp, 3000); // Volta para a tela inicial após 3s
        } else { // verify
            if (result.identified) {
                message.textContent = `Identificado: ${result.name} (Distância: ${result.distance.toFixed(4)})`;
                message.style.color = '#00e676';
                drawResultBox(true);
            } else {
                message.textContent = 'Usuário não identificado.';
                message.style.color = '#ff4d4d';
                drawResultBox(false);
            }
        }
    };

    // Inicia a aplicação
    init();
});

