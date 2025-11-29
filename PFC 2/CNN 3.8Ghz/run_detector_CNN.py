import numpy as np
import time
import threading
import logging
import uhd
import os
import joblib 
import csv
from tensorflow.keras.models import load_model 
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
from scipy import signal
from scipy.fft import fft, fftfreq

# --- Configuração do Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Classes Reutilizadas ---

@dataclass
class X310Config:
    device_args: str = "type=x300"
    rx_rate: float = 10e6
    rx_freq: float = 100e6
    rx_gain: float = 30
    rx_channels: List[int] = None
    rx_antenna: str = "RX2"
    ref_clock: str = "internal"
    cpu_format: str = "fc32"
    wire_format: str = "sc16"
    def __post_init__(self):
        if self.rx_channels is None:
            self.rx_channels = [0]

@dataclass
class SpectrumData:
    frequencies: np.ndarray
    power_dbm: np.ndarray
    center_freq: float
    bandwidth: float
    timestamp: datetime
    fft_size: int

class X310Interface:
    def __init__(self, config: X310Config, simulate: bool = False):
        self.config = config
        self.usrp = None
        self.streamer = None
        self.is_streaming = False
        self.simulate = simulate
        if not self.simulate:
            self._setup_usrp()
        else:
            logger.warning("Rodando em MODO DE SIMULAÇÃO. Nenhum hardware será usado.")

    def _setup_usrp(self):
        try:
            logger.info("Conectando ao USRP X310...")
            self.usrp = uhd.usrp.MultiUSRP(self.config.device_args)
            logger.info(f"Dispositivo conectado: {self.usrp.get_mboard_name()}")
            self.usrp.set_rx_rate(self.config.rx_rate, 0)
            self.usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(self.config.rx_freq), 0)
            self.usrp.set_rx_gain(self.config.rx_gain, 0)
            self.usrp.set_rx_antenna(self.config.rx_antenna, 0)
            self.usrp.set_clock_source(self.config.ref_clock)
            self.usrp.set_time_source(self.config.ref_clock)
            self.usrp.set_time_now(uhd.libpyuhd.types.time_spec(0.0))
            logger.info(f"Taxa de amostragem: {self.usrp.get_rx_rate(0) / 1e6:.1f} MSPS")
            logger.info(f"Frequência central: {self.usrp.get_rx_freq(0) / 1e6:.1f} MHz")
            logger.info(f"Ganho: {self.usrp.get_rx_gain(0):.1f} dB")
        except Exception as e:
            logger.error(f"Erro ao configurar X310: {e}")
            raise

    def create_rx_streamer(self, max_samps_per_packet: int = 1000):
        st_args = uhd.usrp.StreamArgs(self.config.cpu_format, self.config.wire_format)
        st_args.channels = self.config.rx_channels
        self.streamer = self.usrp.get_rx_stream(st_args)
        self.recv_buffer = np.empty(max_samps_per_packet, dtype=np.complex64)
        return self.streamer

    def start_streaming(self):
        if self.simulate:
            self.is_streaming = True
            return
        if not self.streamer:
            self.create_rx_streamer()
        stream_cmd = uhd.libpyuhd.types.stream_cmd(uhd.libpyuhd.types.stream_mode.start_cont)
        stream_cmd.stream_now = True
        self.streamer.issue_stream_cmd(stream_cmd)
        self.is_streaming = True
        logger.info("Streaming iniciado")

    def stop_streaming(self):
        if self.simulate:
            self.is_streaming = False
            return
        if self.streamer and self.is_streaming:
            stream_cmd = uhd.libpyuhd.types.stream_cmd(uhd.libpyuhd.types.stream_mode.stop_cont)
            self.streamer.issue_stream_cmd(stream_cmd)
            self.is_streaming = False
            logger.info("Streaming parado")

    def receive_samples(self, num_samples: int) -> np.ndarray:
        if self.simulate:
            noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * 0.1
            return noise

        if not self.is_streaming:
            raise RuntimeError("Streaming não está ativo")
        
        samples = np.zeros(num_samples, dtype=np.complex64)
        samples_received = 0
        metadata = uhd.libpyuhd.types.rx_metadata()
        
        while samples_received < num_samples:
            samples_to_recv = min(len(self.recv_buffer), num_samples - samples_received)
            samps = self.streamer.recv(self.recv_buffer[:samples_to_recv], metadata, 1.0)
            
            if metadata.error_code != uhd.libpyuhd.types.rx_metadata_error_code.none:
                if metadata.error_code == uhd.libpyuhd.types.rx_metadata_error_code.overflow:
                    continue
                logger.warning(f"Erro na recepção: {metadata.error_code}")
                continue

            samples[samples_received:samples_received + samps] = self.recv_buffer[:samps]
            samples_received += samps
        
        return samples

    def __del__(self):
        if hasattr(self, 'is_streaming') and self.is_streaming:
            self.stop_streaming()

class SpectrumAnalyzer:
    def __init__(self, fft_size: int = 2048, overlap: float = 0.5):
        self.fft_size = fft_size
        self.overlap = overlap
        self.window = signal.windows.hann(fft_size)

    def compute_spectrum(self, samples: np.ndarray, sample_rate: float,
                         center_freq: float) -> SpectrumData:
        if len(samples) < self.fft_size:
            padded_samples = np.zeros(self.fft_size, dtype=samples.dtype)
            padded_samples[:len(samples)] = samples
            samples = padded_samples
        
        segment = samples[:self.fft_size] * self.window
        fft_data = fft(segment)
        freqs = fftfreq(self.fft_size, 1 / sample_rate)

        fft_shifted = np.fft.fftshift(fft_data)
        freqs_shifted = np.fft.fftshift(freqs) + center_freq

        power_watts = np.abs(fft_shifted)**2 / (sample_rate * np.sum(self.window**2))
        power_dbm = 10 * np.log10(power_watts / 1e-3)
        
        return SpectrumData(
            frequencies=freqs_shifted,
            power_dbm=power_dbm,
            center_freq=center_freq,
            bandwidth=sample_rate,
            timestamp=datetime.now(),
            fft_size=self.fft_size
        )

# --- Classe de Detecção CNN ---

class CNNJammerDetector:
    def __init__(self, config: X310Config, 
                 fft_size: int, 
                 model_path: str, 
                 scaler_path: str,
                 alert_cooldown_seconds: float = 3.0,
                 confidence_threshold: float = 75.0,
                 simulate: bool = False):
        
        logger.info("Iniciando o Detector de Jammers (CNN)...")
        self.config = config
        self.fft_size = fft_size
        self.running = False
        
        self.class_names = {0: "Normal (Ocioso)", 1: "Normal (Dados)", 2: "JAMMER"}
        
        self.alert_cooldown_seconds = alert_cooldown_seconds
        self.last_alert_time = 0.0
        self.confidence_threshold = confidence_threshold

        # --- INICIALIZAÇÃO DO CSV ---
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"log_detector_detalhado_{timestamp_str}.csv"
        
        self.csv_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Cabeçalho do CSV (timestamp_abs, tempo_decorrido, id_classe, nome_classe, confianca, status_alerta)
        self.csv_writer.writerow(['timestamp_abs', 'elapsed_time', 'class_id', 'class_name', 'confidence', 'alert_status'])
        
        self.session_start_time = None
        logger.info(f"Log detalhado será salvo em: {self.csv_filename}")
        # ----------------------------

        try:
            logger.info(f"Carregando modelo de: {model_path}")
            self.model = load_model(model_path)
            logger.info(f"Carregando normalizador de: {scaler_path}")
            self.scaler = joblib.load(scaler_path)
        except Exception as e:
            logger.error(f"Erro ao carregar modelo/normalizador: {e}")
            raise

        self.x310 = X310Interface(config, simulate=simulate)
        self.analyzer = SpectrumAnalyzer(fft_size=fft_size)

    def log_event(self, class_id, class_name, confidence, alert_status):
        """
        Escreve uma linha no CSV e garante que seja salva imediatamente.
        """
        if self.session_start_time is None: return
        
        now = time.time()
        elapsed = now - self.session_start_time
        
        line = f"{now:.4f},{elapsed:.4f},{class_id},{class_name},{confidence:.2f},{alert_status}\n"
        
        try:
            self.csv_file.write(line)
            self.csv_file.flush()
        except Exception as e:
            logger.error(f"Erro ao escrever no CSV: {e}")

    def classify_spectrum(self, live_spectrum: SpectrumData) -> (int, float):
        power_data = live_spectrum.power_dbm.reshape(1, -1)
        power_scaled = self.scaler.transform(power_data)
        power_reshaped = power_scaled.reshape((1, self.fft_size, 1))
        prediction = self.model.predict(power_reshaped, verbose=0)
        
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100.0
        
        return predicted_class, confidence

    def run(self):
        samples_per_capture = self.fft_size * 2 
        self.x310.start_streaming()
        self.running = True
        self.session_start_time = time.time()

        logger.info(f"Detector CNN iniciado. Limite de Confiança: {self.confidence_threshold}%")

        try:
            while self.running:
                samples = self.x310.receive_samples(samples_per_capture)
                spectrum = self.analyzer.compute_spectrum(
                    samples, self.config.rx_rate, self.config.rx_freq
                )

                predicted_class, confidence = self.classify_spectrum(spectrum)
                class_name = self.class_names.get(predicted_class, "Desconhecido")
                
                alert_status = "MONITORANDO" # Status padrão para o log

                # --- Lógica de Alerta Filtrada ---
                
                if predicted_class == 2: # Se o modelo acha que é JAMMER
                    
                    if confidence >= self.confidence_threshold:
                        # Só entra aqui se tiver certeza (ex: > 75%)
                        current_time = time.time()
                        if (current_time - self.last_alert_time) > self.alert_cooldown_seconds:
                            self.last_alert_time = current_time
                            
                            alert_status = "ALERTA_JAMMER" # Atualiza status para o log
                            
                            logger.warning("="*40)
                            logger.warning(f"    *** ALERTA DE JAMMER (CNN) ***")
                            logger.warning(f"    Confiança: {confidence:.1f}%")
                            logger.warning("="*40)
                        else:
                            alert_status = "COOLDOWN" # Jammer detectado, mas em cooldown
                    else:
                         # Se a confiança for baixa, ignora
                         if np.random.rand() < 0.2:
                            logger.info(f"Jammer suspeito ignorado (Confiança baixa: {confidence:.1f}%)")
                         alert_status = "IGNORED_LOW_CONF"

                else:
                    # Loga apenas de vez em quando para não poluir o console
                    if np.random.rand() < 0.05:
                        logger.info(f"Classificação: {class_name} (Confiança: {confidence:.1f}%)")
                
                # --- REGISTRO NO CSV (CRÍTICO: Salva tudo o que aconteceu neste ciclo) ---
                self.log_event(predicted_class, class_name, confidence, alert_status)
                
                time.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("Interrupção do usuário recebida. Parando...")
        finally:
            self.stop()
            self.x310.stop_streaming()
            
            # Fecha o arquivo CSV ao sair
            if hasattr(self, 'csv_file') and self.csv_file:
                self.csv_file.close()
                logger.info(f"Arquivo de log FECHADO e SALVO: {self.csv_filename}")

            logger.info("Detector CNN parado.")

    def stop(self):
        self.running = False

def main():
    print("=== Detector de Jammers (baseado em CNN) - v2 ===")

    FFT_SIZE = 4096 
    MODEL_FILE = "jammer_cnn_model5G.h5"
    SCALER_FILE = "spectrum_scaler5G.pkl"

    ALERT_COOLDOWN_SECONDS = 3.0 
    CONFIDENCE_THRESHOLD = 75.0 
    SIMULATE_MODE = False

    x310_config = X310Config(
        device_args="type=x300,addr=192.168.10.2",
        rx_rate=30e6,
        rx_freq=3.75e9,
        rx_gain=40,
        rx_antenna="RX2"
    )

    try:
        detector = CNNJammerDetector(
            config=x310_config,
            fft_size=FFT_SIZE,
            model_path=MODEL_FILE,
            scaler_path=SCALER_FILE,
            alert_cooldown_seconds=ALERT_COOLDOWN_SECONDS,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            simulate=SIMULATE_MODE
        )
        detector.run()

    except FileNotFoundError:
        logger.error(f"Não foi possível iniciar. Modelo ou scaler não encontrados.")
    except Exception as e:
        logger.error(f"Erro inesperado: {e}")

    print("Processo finalizado.")

if __name__ == "__main__":
    main()
