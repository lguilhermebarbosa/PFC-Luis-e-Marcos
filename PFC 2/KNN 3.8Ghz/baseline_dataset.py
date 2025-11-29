#
#
# Sistema de Gravação de Baseline do Espectro Grava um dataset do espectro.
#
#

import numpy as np
import time
import threading
import logging
import uhd
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
from scipy import signal
from scipy.fft import fft, fftfreq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Reutilizando as classes do Detector ---
# classes X310Config, SpectrumData, X310Interface e SpectrumAnalyzer

@dataclass
class X310Config:
    # Configuração do USRP X310
    device_args: str = "type=x300"  # para X310
    rx_rate: float = 10e6  # 10 MSPS
    rx_freq: float = 100e6  # 100 MHz (centro)
    rx_gain: float = 30  # dB
    rx_channels: List[int] = None  # [0] para canal único
    rx_antenna: str = "RX2"  # antena padrão
    ref_clock: str = "internal"  # referência de clock
    cpu_format: str = "fc32"  # formato complexo float32
    wire_format: str = "sc16"  # formato do cabo (otimizado)

    def __post_init__(self):
        if self.rx_channels is None:
            self.rx_channels = [0]

@dataclass
class SpectrumData:
    # dados do espectro que serão analisados
    frequencies: np.ndarray
    power_dbm: np.ndarray
    center_freq: float
    bandwidth: float
    timestamp: datetime
    fft_size: int

class X310Interface:
    # interface do  USRP X310
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
        # configura o USRP X310
        try:
            logger.info("Conectando ao USRP X310...")
            self.usrp = uhd.usrp.MultiUSRP(self.config.device_args)

            # verifica se é um X310
            device_info = self.usrp.get_mboard_name()
            logger.info(f"Dispositivo conectado: {device_info}")

            # configuração básica
            self.usrp.set_rx_rate(self.config.rx_rate, 0)
            self.usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(self.config.rx_freq), 0)
            self.usrp.set_rx_gain(self.config.rx_gain, 0)
            self.usrp.set_rx_antenna(self.config.rx_antenna, 0)

            # configuração de referência
            self.usrp.set_clock_source(self.config.ref_clock)
            self.usrp.set_time_source(self.config.ref_clock)

            # sincronização temporal
            self.usrp.set_time_now(uhd.libpyuhd.types.time_spec(0.0))

            logger.info(f"Taxa de amostragem: {self.usrp.get_rx_rate(0) / 1e6:.1f} MSPS")
            logger.info(f"Frequência central: {self.usrp.get_rx_freq(0) / 1e6:.1f} MHz")
            logger.info(f"Ganho: {self.usrp.get_rx_gain(0):.1f} dB")

        except Exception as e:
            logger.error(f"Erro ao configurar X310: {e}")
            raise

    def create_rx_streamer(self, max_samps_per_packet: int = 1000):
        # cria streamer de recepção
        st_args = uhd.usrp.StreamArgs(self.config.cpu_format, self.config.wire_format)
        st_args.channels = self.config.rx_channels
        self.streamer = self.usrp.get_rx_stream(st_args)

        # buffer de recepção
        self.recv_buffer = np.empty(max_samps_per_packet, dtype=np.complex64)

        return self.streamer

    def start_streaming(self):
        # inicia streaming de dados
        if self.simulate:
            logger.info("Modo de simulação: Pulando início de streaming de hardware.")
            self.is_streaming = True  # streaming está 'ativo' para o loop funcionar
            return

        if not self.streamer:
            self.create_rx_streamer()

        stream_cmd = uhd.libpyuhd.types.stream_cmd(uhd.libpyuhd.types.stream_mode.start_cont)
        stream_cmd.stream_now = True
        self.streamer.issue_stream_cmd(stream_cmd)
        self.is_streaming = True
        logger.info("Streaming iniciado")

    def stop_streaming(self):
        # para streaming
        if self.simulate:
            logger.info("Modo de simulação: Pulando parada de streaming de hardware.")
            self.is_streaming = False
            return

        if self.streamer and self.is_streaming:
            stream_cmd = uhd.libpyuhd.types.stream_cmd(uhd.libpyuhd.types.stream_mode.stop_cont)
            self.streamer.issue_stream_cmd(stream_cmd)
            self.is_streaming = False
            logger.info("Streaming parado")

    def receive_samples(self, num_samples: int) -> np.ndarray:
        # recebe amostras do X310 OU gera dados simulados
        if self.simulate:
            # gera ruído complexo gaussiano como simulação
            noise_power = 0.01
            samples = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * np.sqrt(noise_power / 2)
            # simula um sinal forte para teste
            t = np.arange(num_samples) / self.config.rx_rate
            samples += 0.5 * np.exp(1j * 2 * np.pi * 1e6 * t) # sinal em 1 MHz offset
            return samples

        if not self.is_streaming:
            raise RuntimeError("Streaming não está ativo")

        samples = np.zeros(num_samples, dtype=np.complex64)
        samples_received = 0
        metadata = uhd.libpyuhd.types.rx_metadata()

        while samples_received < num_samples:
            samples_to_recv = min(len(self.recv_buffer), num_samples - samples_received)
            samps = self.streamer.recv(self.recv_buffer[:samples_to_recv], metadata, 1.0)

            if metadata.error_code != uhd.libpyuhd.types.rx_metadata_error_code.none:
                logger.warning(f"Erro na recepção: {metadata.error_code}")
                continue

            samples[samples_received:samples_received + samps] = self.recv_buffer[:samps]
            samples_received += samps

        return samples

    def tune_frequency(self, freq: float):
        # frequência
        self.usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(freq), 0)
        time.sleep(0.1)  # tempo de estabilização

    def set_gain(self, gain: float):
        # ganho
        self.usrp.set_rx_gain(gain, 0)

    def get_device_info(self) -> Dict:
        # retorna informações do dispositivo
        return {
            'device_name': self.usrp.get_mboard_name(),
            'rx_rate': self.usrp.get_rx_rate(0),
            'rx_freq': self.usrp.get_rx_freq(0),
            'rx_gain': self.usrp.get_rx_gain(0),
            'rx_antenna': self.usrp.get_rx_antenna(0),
            'clock_source': self.usrp.get_clock_source(0),
            'time_source': self.usrp.get_time_source(0)
        }

    def __del__(self):
        # cleanup
        if hasattr(self, 'is_streaming') and self.is_streaming:
            self.stop_streaming()


class SpectrumAnalyzer:
    # analisador de espectro
    def __init__(self, fft_size: int = 2048, overlap: float = 0.5):
        self.fft_size = fft_size
        self.overlap = overlap
        self.window = signal.windows.hann(fft_size)
        self.baseline_spectra = {}
        self.max_baseline_history = 50

    def compute_spectrum(self, samples: np.ndarray, sample_rate: float,
                         center_freq: float) -> SpectrumData:
        # calcula espectro de potência
        if len(samples) < self.fft_size:
            # técnica de adicionar zeros ao final ou ao redor de um sinal, para padronizar suas dimensões,
            # melhorar a resolução em transformadas de frequência ou garantir que os dados tenham um tamanho
            # compatível com um algoritmo
            padded_samples = np.zeros(self.fft_size, dtype=samples.dtype)
            padded_samples[:len(samples)] = samples
            samples = padded_samples

        # seleciona segmento
        segment = samples[:self.fft_size] * self.window

        # FFT
        fft_data = fft(segment)

        # frequências
        freqs = fftfreq(self.fft_size, 1 / sample_rate)

        # shift para frequências positivas centralizadas
        fft_shifted = np.fft.fftshift(fft_data)
        freqs_shifted = np.fft.fftshift(freqs) + center_freq

        # potência em dBm (assumindo impedância 50Ω)
        power_watts = np.abs(fft_shifted) ** 2 / (sample_rate * np.sum(self.window ** 2))
        power_dbm = 10 * np.log10(power_watts / 1e-3)  # dBm

        return SpectrumData(
            frequencies=freqs_shifted,
            power_dbm=power_dbm,
            center_freq=center_freq,
            bandwidth=sample_rate,
            timestamp=datetime.now(),
            fft_size=self.fft_size
        )

    def update_baseline(self, spectrum: SpectrumData):
        # atualiza baseline do espectro
        freq_key = spectrum.center_freq

        if freq_key not in self.baseline_spectra:
            self.baseline_spectra[freq_key] = []

        self.baseline_spectra[freq_key].append(spectrum.power_dbm.copy())

        # limita histórico
        if len(self.baseline_spectra[freq_key]) > self.max_baseline_history:
            self.baseline_spectra[freq_key] = self.baseline_spectra[freq_key][-self.max_baseline_history:]

    def get_baseline_spectrum(self, center_freq: float) -> Optional[np.ndarray]:
        # retorna espectro baseline médio
        if center_freq in self.baseline_spectra and self.baseline_spectra[center_freq]:
            return np.mean(self.baseline_spectra[center_freq], axis=0)
        return None

def run_baseline_recorder(config: X310Config, 
                          fft_size: int, 
                          duration_seconds: int, 
                          output_file: str, 
                          simulate: bool = False):

    # função principal para gravar o baseline.

    logger.info(f"Iniciando gravação do baseline por {duration_seconds} segundos...")
    logger.info(f"Os dados serão salvos em: {output_file}")

    try:
        # inicializa a interface do rádio
        x310 = X310Interface(config, simulate=simulate)
        
        # inicializa o analisador de espectro
        analyzer = SpectrumAnalyzer(fft_size=fft_size)

        # lista para guardar todos os espectros capturados
        collected_spectra = []
        
        # define o número de amostras por captura
        # um bom valor é 2x o tamanho da FFT
        samples_per_capture = fft_size * 2

        if not simulate:
            logger.info(f"Dispositivo conectado: {x310.get_device_info()['device_name']}")
        
        x310.start_streaming()
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            try:
                # 1. recebe amostras
                samples = x310.receive_samples(samples_per_capture)

                # 2. calcula o espectro
                spectrum = analyzer.compute_spectrum(
                    samples,
                    config.rx_rate,
                    config.rx_freq
                )

                # 3. armazena o espectro (apenas os dados de potência)
                collected_spectra.append(spectrum.power_dbm)

                if len(collected_spectra) % 10 == 0:
                    logger.info(f"Capturas realizadas: {len(collected_spectra)}")
                time.sleep(0.01) #

            except Exception as e:
                logger.warning(f"Erro na captura: {e}")

    except KeyboardInterrupt:
        logger.info("Gravação interrompida pelo usuário.")
    
    except Exception as e:
        logger.error(f"Erro fatal na gravação: {e}")
        return
    
    finally:
        if 'x310' in locals() and x310.is_streaming:
            x310.stop_streaming()
            logger.info("Streaming parado.")

    # --- salvando o Dataset ---
    if not collected_spectra:
        logger.warning("Nenhum dado foi coletado. O arquivo não será salvo.")
        return

    logger.info(f"Gravação concluída. Total de {len(collected_spectra)} espectros coletados.")
    logger.info(f"Salvando dataset em {output_file}...")

    try:
        # converte a lista para um grande array 2D (linhas=capturas, colunas=bins da FFT)
        dataset_array = np.array(collected_spectra)
        
        # salva o array em um formato NumPy comprimido (.npz)
        # também salva as frequências para referência
        np.savez_compressed(
            output_file, 
            spectra_data=dataset_array, 
            frequencies=spectrum.frequencies # salva o eixo de frequência da última captura
        )
        logger.info("Dataset salvo com sucesso!")

    except Exception as e:
        logger.error(f"Erro ao salvar o arquivo: {e}")


def main():
    """Função principal"""
    print("=== Gravador de Dataset de Baseline Espectral ===")

    # --- configurações ---
    SIMULATE_MODE = False # mudar para False no InComm
    RECORDING_DURATION_SECONDS = 120 # gravação de 2 minutos
    FFT_SIZE = 4096
    OUTPUT_FILE = "normal_baseline_dataset5G.npz" # nome do arquivo

    # configuração do X310
    x310_config = X310Config(
        device_args="type=x300,addr=192.168.10.2",
        rx_rate=40e6,
        rx_freq=3.75e9, # frequência monitorada
        rx_gain=40,
        rx_antenna="RX2"
    )

    run_baseline_recorder(
        config=x310_config,
        fft_size=FFT_SIZE,
        duration_seconds=RECORDING_DURATION_SECONDS,
        output_file=OUTPUT_FILE,
        simulate=SIMULATE_MODE
    )

    print("Processo finalizado.")

if __name__ == "__main__":
    main()
