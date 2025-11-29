import numpy as np
import time
import logging
import uhd
import signal
import sys
from dataclasses import dataclass

# --- Configuração do Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configurações do Hardware ---
@dataclass
class X310Config:
    device_args: str = "type=x300,addr=192.168.10.2" # IP
    rate: float = 20e6         # Taxa de amostragem (ex: 20 MSPS)
    freq: float = 2.44e9       # Frequência central (ex: 2.44 GHz)
    gain: float = 70           # Ganho de TX (Cuidado com a saturação)
    tx_antenna: str = "TX/RX"  # Porta de antena
    cpu_format: str = "fc32"   # Formato no host (float complexo)
    wire_format: str = "sc16"  # Formato no cabo (short complexo - mais eficiente)

class ContinuousJammer:
    def __init__(self, config: X310Config, simulate: bool = False):
        self.config = config
        self.simulate = simulate
        self.usrp = None
        self.tx_streamer = None
        self.noise_buffer = None
        self.running = False

        if not self.simulate:
            self._setup_usrp()
        else:
            logger.warning(">>> MODO SIMULAÇÃO (Sem Hardware) <<<")

        self._generate_noise_buffer()

    def _setup_usrp(self):
        try:
            logger.info("Conectando ao USRP X310...")
            self.usrp = uhd.usrp.MultiUSRP(self.config.device_args)
            logger.info(f"Dispositivo conectado: {self.usrp.get_mboard_name()}")

            self.usrp.set_tx_rate(self.config.rate, 0)
            self.usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(self.config.freq), 0)
            self.usrp.set_tx_gain(self.config.gain, 0)
            self.usrp.set_tx_antenna(self.config.tx_antenna, 0)

            logger.info(f"Taxa TX: {self.usrp.get_tx_rate(0) / 1e6:.2f} MSPS")
            logger.info(f"Freq TX: {self.usrp.get_tx_freq(0) / 1e6:.2f} MHz")
            logger.info(f"Ganho TX: {self.usrp.get_tx_gain(0)} dB")

            st_args = uhd.usrp.StreamArgs(self.config.cpu_format, self.config.wire_format)
            st_args.channels = [0]
            self.tx_streamer = self.usrp.get_tx_stream(st_args)

        except Exception as e:
            logger.error(f"Erro fatal no setup do USRP: {e}")
            sys.exit(1)

    def _generate_noise_buffer(self):
        """Gera um buffer pré-calculado de ruído branco gaussiano."""
        # Gera amostras suficientes para um buffer suave (ex: 10ms de áudio)
        num_samples = int(self.config.rate * 0.01) 
        
        # Ruído complexo (parte real + imaginária)
        noise = np.random.normal(0, 1, num_samples) + 1j * np.random.normal(0, 1, num_samples)
        
        # Normalização para evitar clipping digital (manter amplitude < 1.0)
        max_val = np.max(np.abs(noise))
        if max_val > 0:
            noise = noise / max_val * 0.9 # 0.9 para margem de segurança
            
        self.noise_buffer = noise.astype(np.complex64)
        logger.info(f"Buffer de ruído gerado: {len(self.noise_buffer)} amostras.")

    def start(self):
        self.running = True
        logger.info("========================================")
        logger.info("   JAMMER ATIVO - TRANSMITINDO RUÍDO    ")
        logger.info("========================================")

        if self.simulate:
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            return

        # Configuração de metadados para streaming
        metadata = uhd.libpyuhd.types.tx_metadata()
        metadata.start_of_burst = True
        metadata.end_of_burst = False
        metadata.has_time_spec = False

        try:
            while self.running:
                # Envia o buffer repetidamente
                samples_sent = self.tx_streamer.send(self.noise_buffer, metadata)
                
                # Após o primeiro pacote, não é mais o início do burst
                metadata.start_of_burst = False

                if samples_sent == 0:
                    logger.warning("Nenhuma amostra enviada (Timeout?)")
                    
        except KeyboardInterrupt:
            logger.info("Interrupção recebida...")
        except Exception as e:
            logger.error(f"Erro durante transmissão: {e}")
        finally:
            # Envia flag de fim de burst para fechar o stream corretamente
            logger.info("Encerrando transmissão...")
            metadata.end_of_burst = True
            self.tx_streamer.send(np.zeros(10, dtype=np.complex64), metadata)

    def stop(self, signum=None, frame=None):
        self.running = False
        logger.info("Parando serviço...")

# --- Função Principal ---
def main():
    # Configurações Diretas
    config = X310Config(
        device_args="type=x300,addr=192.168.40.2", # IP
        rate=60e6,    # 20 MHz de largura de banda
        freq=3.7e9,  # 2.44 GHz
        gain=75       # Potência
    )

    jammer = ContinuousJammer(config, simulate=False)

    signal.signal(signal.SIGINT, jammer.stop)
    signal.signal(signal.SIGTERM, jammer.stop)

    # FOGO!
    jammer.start()

if __name__ == "__main__":
    main()