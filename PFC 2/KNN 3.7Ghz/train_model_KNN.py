#
# TREINAMENTO DE MODELO USANDO KNN (K-Nearest Neighbors)
#
# Dependências: pip install scikit-learn joblib numpy
#

import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constantes ---
FILE_NORMAL = "normal_baseline_dataset5G.npz"
FILE_DADOS = "dados_baseline_dataset5G.npz"
FILE_JAMMER = "jammer_baseline_dataset5G.npz"

# Alterado para .pkl pois o KNN é salvo via joblib, não Keras
MODEL_FILENAME = "jammer_knn_model5G.pkl" 
SCALER_FILENAME = "spectrum_scaler5G.pkl"

FFT_SIZE = 4096 

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
    Normaliza os dados para o KNN.
    Diferente da CNN, não fazemos reshape para 3D nem to_categorical nos labels.
    """
    logger.info("Pré-processando dados (Normalização)...")
    
    # 1. Normalização (CRÍTICO para KNN pois ele calcula distâncias)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # NOTA: Não fazemos reshape para (amostras, fft, 1). 
    # O KNN precisa de (amostras, features), que já é o formato atual.
    
    logger.info("Dados prontos para o treino.")
    
    # Salva o 'scaler' para usá-lo na detecção ao vivo
    joblib.dump(scaler, SCALER_FILENAME)
    logger.info(f"Normalizador salvo em: {SCALER_FILENAME}")
    
    return X_scaled, y # Retorna y direto (0, 1, 2), sem One-Hot Encoding

def train_knn(X_train, y_train, n_neighbors=5):
    """
    Treina o classificador KNN.
    """
    logger.info(f"Treinando KNN com K={n_neighbors}...")
    # n_jobs=-1 usa todos os processadores disponíveis para calcular distâncias
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(X_train, y_train)
    return knn

def main():
    X, y = load_data()
    if X is None:
        return

    X_processed, y_processed = preprocess_data(X, y)
    
    # Divide em dados de treino (80%) e teste (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42
    )

    logger.info("--- Iniciando Treinamento KNN ---")
    
    # Treina o modelo
    # K=5 é um bom padrão. Se houver muito ruído, aumente (ex: 7 ou 9).
    model = train_knn(X_train, y_train, n_neighbors=5)
    
    logger.info("--- Treinamento Concluído. Avaliando... ---")

    # Avalia o modelo final
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Acurácia no teste: {accuracy * 100:.2f}%")
    logger.info("\nRelatório de Classificação:\n" + classification_report(y_test, y_pred, target_names=['Normal', 'Dados', 'Jammer']))

    # Salva o modelo treinado usando joblib
    joblib.dump(model, MODEL_FILENAME)
    logger.info(f"Modelo salvo em: {MODEL_FILENAME}")
    logger.info("IMPORTANTE: Lembre-se de atualizar seu script de detecção para carregar .pkl com joblib!")

if __name__ == "__main__":
    main()