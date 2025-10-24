import tensorflow as tf # type: ignore
import numpy as np # type: ignore
from PIL import Image # type: ignore
import io
import base64
import albumentations as A # type: ignore
from .model_class import *


# --- Constantes ---
IMG_SIZE = 224

def load_keras_model(model_path='model_v1.keras'):
    """
    Carrega o modelo Keras a partir do caminho especificado.
    Retorna o modelo carregado ou None em caso de erro.
    """
    try:
        print(f"Carregando modelo de: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("Modelo carregado com sucesso.")
        # Opcional: faz uma predição "dummy" para aquecer o modelo
        dummy_input = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        model.feature_extractor.predict(dummy_input)
        print("Modelo aquecido.")
        return model
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None

def preprocess_image(base64_string: str) -> np.ndarray:
    """
    Decodifica uma string base64, pré-processa a imagem e a prepara para o modelo.
    
    Args:
        base64_string: A imagem codificada como uma string base64.

    Returns:
        Um tensor numpy (IMG_SIZE, IMG_SIZE, 3) uint8.
    """
    # Remove o cabeçalho do base64 (ex: "data:image/jpeg;base64,")    
    if "," in base64_string: # TODO: check base64
        base64_string = base64_string.split(',')[1]            

    # Decodifica a string base64 para bytes
    img_bytes = base64.b64decode(base64_string)    

    # Abre a imagem usando Pillow
    img = Image.open(io.BytesIO(img_bytes))    
    
    # Converte para RGB (caso seja RGBA, por exemplo)
    if img.mode != 'RGB':
        img = img.convert('RGB')    
        
    # Redimensiona a imagem
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    # Converte a imagem para um array numpy
    img_array = tf.keras.preprocessing.image.img_to_array(img).astype('uint8')
    
    return img_array # (224, 224, 3)

# Support Augmentation transform
transform = A.Compose([
    A.RandomOrder([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-15, 15), shear=(-5, 5), border_mode=1, p=0.7),
        A.RGBShift((-10, 10), (-10, 10), (-10, 10)),
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.75),
        A.OpticalDistortion(distort_limit=(-0.1, 0.1), p=0.75),
        A.Perspective(scale=(0.01, 0.05), p=0.75),        
    ], n=4)
])
augment = lambda img: transform(image=img)['image']
