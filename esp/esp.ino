#include <driver/dac.h>
#include <HardwareSerial.h>

// HARDWARE CONSTANTS
// Changed SPEAKER_PIN to 26 (DAC2) because 25 is used for Red LED.
#define SPEAKER_PIN 26 

// LED INDICATORS
#define RED_LED_PIN 25   // Indicates Recording
#define WHITE_LED_PIN 27 // Indicates Playback

// INPUTS
#define MIC_PIN 34    // Microphone Analog Input
#define BUTTON_PIN 4  // Push Button (Internal Pull-up, Active LOW)

// AUDIO SETTINGS
#define SAMPLE_RATE 16000
#define BAUD_RATE 500000 
#define BUFFER_SIZE 4096 

// CIRCULAR BUFFER
volatile uint8_t audioBuffer[BUFFER_SIZE];
volatile int head = 0;
volatile int tail = 0;

// TIMER
hw_timer_t * timer = NULL;
portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;

// INTERRUPT SERVICE ROUTINE
void IRAM_ATTR onTimer() {
  portENTER_CRITICAL_ISR(&timerMux);
  
  // If buffer has data, play it
  if (head != tail) {
     // PLAYBACK MODE
     digitalWrite(WHITE_LED_PIN, HIGH); // Turn ON White LED

     uint8_t val = audioBuffer[tail];
     tail = (tail + 1) % BUFFER_SIZE;
     
     // Write to DAC (0-255)
     dacWrite(SPEAKER_PIN, val);
  } else {
     // IDLE / SILENCE
     digitalWrite(WHITE_LED_PIN, LOW); // Turn OFF White LED

     // Buffer empty, output silence (midpoint) to reduce popping
     dacWrite(SPEAKER_PIN, 128);
  }
  
  portEXIT_CRITICAL_ISR(&timerMux);
}

void setup() {
  Serial.begin(BAUD_RATE);
  
  // Configure Pins
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(WHITE_LED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP); // Button to GND, uses internal resistor
  pinMode(MIC_PIN, INPUT);
  
  // Default State
  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(WHITE_LED_PIN, LOW);
  
  // Setup Timer for 16kHz playback
  // timerBegin(frequency) is for Arduino ESP32 v3.0+
  // Ensure we use a compatible timer setup
  timer = timerBegin(1000000); 
  timerAttachInterrupt(timer, &onTimer);
  timerAlarm(timer, 1000000 / SAMPLE_RATE, true, 0); 
}

// Global variables for timing and state
unsigned long lastSampleTime = 0;
const unsigned long sampleInterval = 62; // 16kHz ~= 62.5us
bool isRecording = false;

void loop() {
  // --- RECORDING LOGIC ---
  bool btnPressed = (digitalRead(BUTTON_PIN) == LOW);
  
  if (btnPressed) {
    if (!isRecording) {
        isRecording = true;
        digitalWrite(RED_LED_PIN, HIGH); // Turn ON Red LED only once
    }
    
    unsigned long now = micros();
    if (now - lastSampleTime >= sampleInterval) {
        lastSampleTime = now;
        
        // Read Microphone
        int micValue = analogRead(MIC_PIN); // 0-4095
        uint8_t audioByte = map(micValue, 0, 4095, 0, 255);
        Serial.write(audioByte);
    }
    
  } else {
    if (isRecording) {
        isRecording = false;
        digitalWrite(RED_LED_PIN, LOW); // Turn OFF Red LED
    }
  }

  // --- PLAYBACK DATA RECEPTION ---
  // Read data from Serial into Buffer if available
  // We process this even during recording phase to keep buffer behavior sanity,
  // though typically we don't full duplex here.
  
  while (Serial.available()) {
    uint8_t byte = Serial.read();
    
    int nextHead = (head + 1) % BUFFER_SIZE;
    
    // If buffer not full, add to buffer
    if (nextHead != tail) {
        portENTER_CRITICAL(&timerMux);
        audioBuffer[head] = byte;
        head = nextHead;
        portEXIT_CRITICAL(&timerMux);
    } 
  }
}
