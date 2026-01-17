
# lembrar de usar o pip install tensorflow scikit-learn joblib

import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib # Para salvar o "normalizador"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constantes ---
FILE_NORMAL = "normal_baseline_dataset5G.npz"
FILE_DADOS = "dados_baseline_dataset5G.npz"
FILE_JAMMER = "jammer_baseline_dataset5G.npz"

MODEL_FILENAME = "jammer_cnn_model5G.h5"
SCALER_FILENAME = "spectrum_scaler5G.pkl"

FFT_SIZE = 4096 # DEVE ser o mesmo FFT_SIZE dos seus datasets
NUM_CLASSES = 3

def load_data():
    """
    Carrega os 3 datasets e cria os 'labels' (etiquetas).
    Classe 0: Normal
    Classe 1: Dados
    Classe 2: Jammer
    """
    global FFT_SIZE

    try:
        logger.info("Carregando datasets...")
        
        # Carrega os dados de espectro
        normal_data = np.load(FILE_NORMAL)['spectra_data']
        dados_data = np.load(FILE_DADOS)['spectra_data']
        jammer_data = np.load(FILE_JAMMER)['spectra_data']
        
        logger.info(f"Amostras 'Normal': {len(normal_data)}")
        logger.info(f"Amostras 'Dados': {len(dados_data)}")
        logger.info(f"Amostras 'Jammer': {len(jammer_data)}")

        # Cria os labels (etiquetas)
        # 0 = normal, 1 = dados, 2 = jammer
        normal_labels = np.zeros(len(normal_data))
        dados_labels = np.ones(len(dados_data))
        jammer_labels = np.full(len(jammer_data), 2)

        # Junta tudo em dois arrays grandes: X (dados) e y (labels)
        X = np.concatenate((normal_data, dados_data, jammer_data), axis=0)
        y = np.concatenate((normal_labels, dados_labels, jammer_labels), axis=0)
        
        # Embaralha os dados para o treino
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        # Validação de FFT
        if X.shape[1] != FFT_SIZE:
             logger.warning(f"Tamanho de FFT inconsistente! Esperado {FFT_SIZE}, encontrado {X.shape[1]}")
             # Agora a modificação da variável global é válida
             FFT_SIZE = X.shape[1] 
             logger.info(f"Tamanho da FFT ajustado para {FFT_SIZE}")


        return X, y

    except FileNotFoundError as e:
        logger.error(f"Erro: Arquivo não encontrado: {e.filename}")
        return None, None
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        return None, None

def preprocess_data(X, y):
    """
    Normaliza e formata os dados para a CNN.
    """
    logger.info("Pré-processando dados...")
    
    # 1. Normalização (CRÍTICO para CNNs)
    # Os dados (dBm) precisam ficar entre 0 e 1.
    # Usamos o MinMaxScaler, que aprende o min/max de *cada bin* de frequência.
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Reshape para a CNN 1D
    # A CNN (Conv1D) espera [amostras, passos_de_tempo, canais]
    # Nossos dados são [amostras, bins_fft]
    # Vamos formatar para [amostras, bins_fft, 1] (1 canal)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    
    # 3. Converte labels para formato "categórico"
    # (ex: 2 -> [0, 0, 1])
    y_cat = to_categorical(y, num_classes=NUM_CLASSES)
    
    logger.info("Dados prontos para o treino.")
    
    joblib.dump(scaler, SCALER_FILENAME)
    logger.info(f"Normalizador salvo em: {SCALER_FILENAME}")
    
    return X_reshaped, y_cat

def build_model(input_shape):
    """
    Cria a arquitetura da CNN 1D.
    """
    logger.info("Construindo modelo CNN...")
    model = Sequential()
    
    # Camada Convolucional 1
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    
    # Camada Convolucional 2
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    # Achata os dados para a camada Densa (classificação)
    model.add(Flatten())
    
    # Camada Densa
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5)) # Camada de Dropout para evitar overfitting
    
    # Camada de Saída
    model.add(Dense(NUM_CLASSES, activation='softmax')) # 'softmax' para classificação de múltiplas classes
    
    # Compila o modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    return model

def main():
    X, y = load_data()
    if X is None:
        return

    X_processed, y_processed = preprocess_data(X, y)
    
    # Divide em dados de treino (80%) e teste (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42
    )

    input_shape = (FFT_SIZE, 1) # (4096, 1)
    model = build_model(input_shape)

    logger.info("--- Iniciando Treinamento ---")
    
    # Treina o modelo
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,       # 10 "passadas" pelos dados. Pode precisar de mais.
        batch_size=32    # Processa em lotes de 32 amostras
    )
    
    logger.info("--- Treinamento Concluído ---")

    loss, accuracy = model.evaluate(X_test, y_test)
    logger.info(f"Acurácia no teste: {accuracy * 100:.2f}%")

    model.save(MODEL_FILENAME)
    logger.info(f"Modelo salvo em: {MODEL_FILENAME}")

if __name__ == "__main__":
    main()