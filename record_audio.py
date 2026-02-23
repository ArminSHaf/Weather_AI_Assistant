import serial
import wave
import serial.tools.list_ports
import time
import sys

def get_serial_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No COM ports found!")
        return None
    
    print("Available ports:")
    for i, p in enumerate(ports):
        print(f"{i}: {p.device} - {p.description}")
        
    if len(ports) == 1:
        return ports[0].device
        
    try:
        idx = int(input("Select port index: "))
        return ports[idx].device
    except:
        return None

def main():
    print("--- AUDIO RECORDER ---")
    port = get_serial_port()
    if not port:
        return

    # Must match ESP32 settings
    BAUD_RATE = 500000 
    SAMPLE_RATE = 16000
    OUTPUT_FILE = "recorded_audio.wav"

    try:
        ser = serial.Serial(port, BAUD_RATE)
        print(f"Connected to {port}. Ready to record.")
    except Exception as e:
        print(f"Error opening serial: {e}")
        return

    # Open WAV file for writing
    # 1 channel (mono), 1 byte per sample (8-bit), 16000Hz
    wf = wave.open(OUTPUT_FILE, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(1) # 8-bit
    wf.setframerate(SAMPLE_RATE)

    print("\nINSTRUCTIONS:")
    print("1. Hold the EXTERNAL PUSH BUTTON (GPIO 4) to record.")
    print("2. The RED LED (GPIO 25) will turn ON while recording.")
    print("3. Release to pause.")
    print("4. Press Ctrl+C here to STOP and SAVE the file.")
    print("\nRecording... (Silence is skipped if button not pressed)")

    try:
        total_samples = 0
        while True:
            # Read all available bytes
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                wf.writeframes(data)
                total_samples += len(data)
                sys.stdout.write(f"\rCaptured: {total_samples/SAMPLE_RATE:.1f} seconds")
                sys.stdout.flush()
            
            # Sleep slightly to let buffer fill
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nStopping...")
        
    wf.close()
    ser.close()
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
