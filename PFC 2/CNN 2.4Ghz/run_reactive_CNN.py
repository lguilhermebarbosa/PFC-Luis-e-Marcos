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

# --- Configurações ---

@dataclass
class X310Config:
    device_args: str = "type=x300"
    rx_rate: float = 10e6
    rx_freq: float = 100e6
    rx_gain: float = 30
    tx_gain: float = 60 
    rx_channels: List[int] = None
    rx_antenna: str = "RX2"
    tx_antenna: str = "TX/RX" 
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
        self.rx_streamer = None
        self.tx_streamer = None 
        self.is_streaming = False
        self.simulate = simulate
        
        self.attack_buffer = None

        if not self.simulate:
            self._setup_usrp()
            self._prepare_attack_buffer()
        else:
            logger.warning("Rodando em MODO DE SIMULAÇÃO. Nenhum hardware será usado.")

    def _setup_usrp(self):
        try:
            logger.info("Conectando ao USRP X310...")
            self.usrp = uhd.usrp.MultiUSRP(self.config.device_args)
            logger.info(f"Dispositivo conectado: {self.usrp.get_mboard_name()}")

            # --- Configuração RX ---
            self.usrp.set_rx_rate(self.config.rx_rate, 0)
            self.usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(self.config.rx_freq), 0)
            self.usrp.set_rx_gain(self.config.rx_gain, 0)
            self.usrp.set_rx_antenna(self.config.rx_antenna, 0)
            
            # --- Configuração TX ---
            self.usrp.set_tx_rate(self.config.rx_rate, 0) 
            self.usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(self.config.rx_freq), 0) 
            self.usrp.set_tx_gain(self.config.tx_gain, 0)
            self.usrp.set_tx_antenna(self.config.tx_antenna, 0)

            # Configuração de Clock
            self.usrp.set_clock_source(self.config.ref_clock)
            self.usrp.set_time_source(self.config.ref_clock)
            self.usrp.set_time_now(uhd.libpyuhd.types.time_spec(0.0))

            logger.info(f"Taxa: {self.usrp.get_rx_rate(0) / 1e6:.1f} MSPS")
            logger.info(f"Freq: {self.usrp.get_rx_freq(0) / 1e6:.1f} MHz")
            logger.info(f"Ganho RX: {self.config.rx_gain} dB | Ganho TX: {self.config.tx_gain} dB")

        except Exception as e:
            logger.error(f"Erro ao configurar X310: {e}")
            raise

    def _prepare_attack_buffer(self):
        num_samples = int(self.config.rx_rate * 0.01) 
        noise = np.random.normal(0, 1, num_samples) + 1j * np.random.normal(0, 1, num_samples)
        max_val = np.max(np.abs(noise))
        if max_val > 0:
            noise = noise / max_val * 0.7
        self.attack_buffer = noise.astype(np.complex64)
        logger.info(f"Buffer de ataque gerado: {len(self.attack_buffer)} amostras.")

    def create_streamers(self, max_samps_per_packet: int = 1000):
        st_args = uhd.usrp.StreamArgs(self.config.cpu_format, self.config.wire_format)
        st_args.channels = self.config.rx_channels
        self.rx_streamer = self.usrp.get_rx_stream(st_args)
        self.recv_buffer = np.empty(max_samps_per_packet, dtype=np.complex64)
        self.tx_streamer = self.usrp.get_tx_stream(st_args)
        return self.rx_streamer

    def fire_jammer(self, duration_cycles=10):
        if self.simulate:
            logger.warning(">>> [SIMULAÇÃO] JAMMER DISPARADO! <<<")
            return
        if self.tx_streamer is None or self.attack_buffer is None:
            return
        metadata = uhd.libpyuhd.types.tx_metadata()
        metadata.start_of_burst = True
        metadata.end_of_burst = False
        metadata.has_time_spec = False
        try:
            for i in range(duration_cycles):
                if i == duration_cycles - 1:
                    metadata.end_of_burst = True
                self.tx_streamer.send(self.attack_buffer, metadata)
                metadata.start_of_burst = False 
        except Exception as e:
            logger.error(f"Erro no disparo do Jammer: {e}")

    def start_rx_streaming(self):
        if self.simulate:
            self.is_streaming = True
            return
        if not self.rx_streamer:
            self.create_streamers()
        stream_cmd = uhd.libpyuhd.types.stream_cmd(uhd.libpyuhd.types.stream_mode.start_cont)
        stream_cmd.stream_now = True
        self.rx_streamer.issue_stream_cmd(stream_cmd)
        self.is_streaming = True
        logger.info("Streaming RX iniciado")

    def stop_rx_streaming(self):
        if self.simulate:
            self.is_streaming = False
            return
        if self.rx_streamer and self.is_streaming:
            stream_cmd = uhd.libpyuhd.types.stream_cmd(uhd.libpyuhd.types.stream_mode.stop_cont)
            self.rx_streamer.issue_stream_cmd(stream_cmd)
            self.is_streaming = False
            logger.info("Streaming RX parado")

    def receive_samples(self, num_samples: int) -> np.ndarray:
        if self.simulate:
            noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * 0.1
            if np.random.rand() < 0.3: 
                return noise + 0.5 
            return noise

        if not self.is_streaming:
            raise RuntimeError("Streaming não está ativo")
        
        samples = np.zeros(num_samples, dtype=np.complex64)
        samples_received = 0
        metadata = uhd.libpyuhd.types.rx_metadata()
        
        while samples_received < num_samples:
            samples_to_recv = min(len(self.recv_buffer), num_samples - samples_received)
            samps = self.rx_streamer.recv(self.recv_buffer[:samples_to_recv], metadata, 1.0)
            
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
            self.stop_rx_streaming()

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
        power_dbm = 10 * np.log10(power_watts / 1e-3 + 1e-15)
        return SpectrumData(
            frequencies=freqs_shifted,
            power_dbm=power_dbm,
            center_freq=center_freq,
            bandwidth=sample_rate,
            timestamp=datetime.now(),
            fft_size=self.fft_size
        )

# --- Classe Reativa ---

class ReactiveJammerBot:
    def __init__(self, config: X310Config, 
                 fft_size: int, 
                 model_path: str, 
                 scaler_path: str,
                 attack_cooldown: float = 2.0,
                 confidence_threshold: float = 85.0,
                 simulate: bool = False):
        
        logger.info("Iniciando SISTEMA REATIVO (Jammer Automático)...")
        self.config = config
        self.fft_size = fft_size
        self.running = False
        
        # Mapeamento de classes
        self.class_names = {0: "Normal (Ocioso)", 1: "ALVO: DADOS", 2: "Jammer (Eu/Outro)"}
        
        self.attack_cooldown = attack_cooldown
        self.last_attack_time = 0.0
        self.confidence_threshold = confidence_threshold

        # --- INICIALIZAÇÃO DO CSV ---
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"log_sessao_detalhado_{timestamp_str}.csv"
        
        self.csv_file = open(self.csv_filename, 'w', newline='')
        
        # Escreve o cabeçalho
        # timestamp_abs: Tempo Unix (para plotar)
        # elapsed: Tempo desde inicio (segundos)
        # class_id: 0, 1 ou 2
        # class_name: Nome legível
        # confidence: Certeza da IA
        # action: O que o bot fez (MONITORANDO, ATAQUE_INICIADO, ATAQUE_FIM, COOLDOWN)
        self.csv_file.write("timestamp_abs,elapsed_time,class_id,class_name,confidence,action\n")
        
        self.session_start_time = None
        logger.info(f"Log detalhado será salvo em: {self.csv_filename}")
        # ----------------------------

        try:
            logger.info(f"Carregando cérebro de: {model_path}")
            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)
        except Exception as e:
            logger.error(f"Erro fatal ao carregar IA: {e}")
            raise

        self.x310 = X310Interface(config, simulate=simulate)
        self.analyzer = SpectrumAnalyzer(fft_size=fft_size)

    def log_event(self, class_id, class_name, confidence, action):
        """
        Escreve uma linha no CSV e garante que seja salva imediatamente.
        """
        if self.session_start_time is None: return
        
        now = time.time()
        elapsed = now - self.session_start_time
        
        line = f"{now:.4f},{elapsed:.4f},{class_id},{class_name},{confidence:.2f},{action}\n"
        
        try:
            self.csv_file.write(line)
            self.csv_file.flush() # Força a escrita no disco (CRÍTICO para não perder dados no Ctrl+C)
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
        self.x310.start_rx_streaming()
        self.running = True
        self.session_start_time = time.time() # Marca início da sessão
        
        logger.info("==========================================")
        logger.info(f" VIGILÂNCIA ATIVA INICIADA")
        logger.info(f" ALVO: Classe 1 (Dados)")
        logger.info(f" GATILHO: Confiança > {self.confidence_threshold}%")
        logger.info("==========================================")

        try:
            while self.running:
                samples = self.x310.receive_samples(samples_per_capture)
                spectrum = self.analyzer.compute_spectrum(
                    samples, self.config.rx_rate, self.config.rx_freq
                )

                predicted_class, confidence = self.classify_spectrum(spectrum)
                class_name = self.class_names[predicted_class]
                
                # Variável para registrar o estado atual no CSV
                current_action = "MONITORANDO"

                # --- LÓGICA DE ATAQUE REATIVO ---
                if predicted_class == 1 and confidence >= self.confidence_threshold:
                    
                    current_time = time.time()
                    # Verifica se passaram 5 segundos DESDE O FIM do último ataque
                    if (current_time - self.last_attack_time) > self.attack_cooldown:
                        
                        # 1. Registra o momento exato ANTES do ataque
                        self.log_event(predicted_class, class_name, confidence, "ATAQUE_INICIADO")
                        
                        logger.warning("🛑 TRÁFEGO DE DADOS DETECTADO! INICIANDO JAMMER...")
                        logger.warning(f"   Confiança do Alvo: {confidence:.1f}%")
                        
                        # Ação de Bloqueio
                        self.x310.fire_jammer(duration_cycles=1000)
                        
                        logger.info("   Ataque concluído. Iniciando timer de 5s...")
                        
                        # Atualiza relógio APÓS ataque
                        self.last_attack_time = time.time()
                        
                        # 2. Registra o momento exato DEPOIS do ataque
                        self.log_event(predicted_class, class_name, confidence, "ATAQUE_FINALIZADO")
                        
                        time.sleep(0.1) 
                        continue # Pula o log padrão abaixo para não duplicar
                    
                    else:
                        # Detectou alvo, mas está esperando o timer
                        current_action = "COOLDOWN"
                
                elif predicted_class == 0:
                    if np.random.rand() < 0.05: 
                        logger.info(f"Rede Limpa/Ociosa ({confidence:.1f}%)")
                
                # --- REGISTRO NO CSV (Para todos os ciclos normais) ---
                self.log_event(predicted_class, class_name, confidence, current_action)
                
                time.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("Interrupção do usuário recebida.")
        finally:
            self.stop()
            self.x310.stop_rx_streaming()
            
            # Fecha o arquivo CSV
            if hasattr(self, 'csv_file') and self.csv_file:
                self.csv_file.close()
                logger.info(f"Arquivo de log FECHADO e SALVO: {self.csv_filename}")
            
            logger.info("Sistema parado.")

    def stop(self):
        self.running = False

def main():
    # --- CONFIGURAÇÕES DE ATAQUE ---
    FFT_SIZE = 4096 
    MODEL_FILE = "jammer_cnn_model.h5"
    SCALER_FILE = "spectrum_scaler.pkl"

    ATTACK_COOLDOWN = 5.0 
    CONFIDENCE_THRESHOLD = 97.0 
    
    SIMULATE_MODE = False

    # Configuração do Hardware
    x310_config = X310Config(
        device_args="type=x300,addr=192.168.40.2",
        rx_rate=80e6,   
        rx_freq=2.44e9, 
        rx_gain=40,    
        tx_gain=60,    
        rx_antenna="RX2",
        tx_antenna="TX/RX" 
    )

    try:
        bot = ReactiveJammerBot(
            config=x310_config,
            fft_size=FFT_SIZE,
            model_path=MODEL_FILE,
            scaler_path=SCALER_FILE,
            attack_cooldown=ATTACK_COOLDOWN,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            simulate=SIMULATE_MODE
        )
        bot.run()

    except FileNotFoundError:
        logger.error(f"Arquivos do modelo não encontrados.")
    except Exception as e:
        logger.error(f"Erro inesperado: {e}")

    print("Processo finalizado.")

if __name__ == "__main__":
    main()