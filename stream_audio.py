import serial
import wave
import time
import struct
import sys
import os
import serial.tools.list_ports

# Configuration
# Resolves to: d:/an3/sem1/IOT/proiect/Weather_AI_Assistant-main/audio_folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_FOLDER = os.path.join(BASE_DIR, "Weather_AI_Assistant-main", "audio_folder")
INPUT_FILE = os.path.join(AUDIO_FOLDER, "audio.wav")   # ESP32 -> PC
OUTPUT_FILE = os.path.join(AUDIO_FOLDER, "reply.wav")  # PC -> ESP32

BAUD_RATE = 500000
SAMPLE_RATE = 16000
CHANNELS = 1
WIDTH = 1 # Bytes per sample (8-bit form ESP32)

def get_serial_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No COM ports found!")
        return None
    
    # Auto-select if only one
    if len(ports) == 1:
        print(f"Auto-selecting {ports[0].device}")
        return ports[0].device

    print("Available ports:")
    for i, p in enumerate(ports):
        print(f"{i}: {p.device} - {p.description}")
        
    try:
        idx = int(input("Select port index: "))
        return ports[idx].device
    except:
        return None

class AudioBridge:
    def __init__(self, port):
        self.ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
        self.last_reply_mtime = 0
        
        # Ensure audio folder exists
        if not os.path.exists(AUDIO_FOLDER):
            os.makedirs(AUDIO_FOLDER)
            
        print(f"Connected to {port} at {BAUD_RATE}")
        print(f"Monitoring {AUDIO_FOLDER}...")

    def listen(self):
        print("\nListening for incoming audio from ESP32... (Press Ctrl+C to stop)")
        try:
            while True:
                try:
                    # 1. Check if ESP32 is sending data (Recording)
                    if self.ser.in_waiting > 0:
                        self.record_stream()
                except serial.SerialException as e:
                    print(f"Serial Error: {e}. Retrying in 1s...")
                    time.sleep(1)
                    continue
                except OSError as e:
                    print(f"OS Error: {e}. Retrying...")
                    time.sleep(1)
                    continue
                
                # 2. Check if there is a new reply to play
                if os.path.exists(OUTPUT_FILE):
                    try:
                        mtime = os.path.getmtime(OUTPUT_FILE)
                        # Debounce: Ensure file is at least 1 second newer than last play
                        if mtime > self.last_reply_mtime + 0.5:
                            
                            # Double check it hasn't changed in the last 100ms (write finished)
                            time.sleep(0.2)
                            mtime2 = os.path.getmtime(OUTPUT_FILE)
                            if mtime2 != mtime:
                                continue # Still writing

                            print(f"\nNew reply detected! ({OUTPUT_FILE})")
                            self.play_file(OUTPUT_FILE)
                            self.last_reply_mtime = mtime
                    except Exception as e:
                        print(f"Error checking file: {e}")
                
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.ser.close()

    def record_stream(self):
        print("\nRecording started...", end="")
        frames = bytearray()
        
        # Timeout variables to detect silence/end of transmission
        last_data_time = time.time()
        timeout = 0.5 # Seconds of silence to consider recording done
        
        while True:
            if self.ser.in_waiting > 0:
                chunk = self.ser.read(self.ser.in_waiting)
                frames.extend(chunk)
                last_data_time = time.time()
                # progress indicator
                if len(frames) % 4000 == 0:
                    print(".", end="", flush=True)
            else:
                # No data waiting, check timeout
                if time.time() - last_data_time > timeout:
                    break
                time.sleep(0.005)
                
        print(f"\nRecording finished. captured {len(frames)} bytes.")
        
        # Save to WAV
        try:
            with wave.open(INPUT_FILE, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(WIDTH)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(frames)
            print(f"Saved to {INPUT_FILE}")
        except Exception as e:
            print(f"Error saving file: {e}")

    def play_file(self, filename):
        print(f"Playing {filename}...")
        try:
            wf = wave.open(filename, 'rb')
        except wave.Error:
            print(f"Error: {filename} is not a valid WAVE file. (Is it MP3?)")
            return
        except FileNotFoundError:
            print("File not found.")
            return

        # Audio parameters
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        
        chunk_size = 1024
        target_rate = 16000 # ESP32 playback rate
        
        # Rudimentary resampling/skipping if rate is different (ESP32 expects ~16k)
        # Assuming 16kHz for now or handling speedup by simple skip
        skip = 1
        if rate > target_rate:
            skip = int(rate / target_rate)
            
        # Volume/Normalization could go here
        
        data = wf.readframes(chunk_size)
        total_sent = 0
        
        while len(data) > 0:
            start_t = time.time()
            
            # Prepare data for ESP32 (Single byte, raw)
            # This logic assumes the WAV is already roughly compatible or needs simple conversion
            # For robustness, we might want to just send the raw bytes if it matches.
            
            output_chunk = bytearray()
            
            # Simple conversion loop
            step = width * channels
            for i in range(0, len(data), step * skip):
                if i+step > len(data): break
                
                # Resample logic (very basic)
                sample_val = 0
                if width == 1:
                    sample_val = data[i]
                elif width == 2:
                    # 16-bit signed to 8-bit unsigned
                    val = struct.unpack('<h', data[i:i+2])[0]
                    sample_val = (val + 32768) >> 8
                
                output_chunk.append(sample_val)
                
            self.ser.write(output_chunk)
            total_sent += len(output_chunk)
            
            # Flow control
            # bytes / rate = duration
            duration = len(output_chunk) / target_rate
            elapsed = time.time() - start_t
            if duration > elapsed:
                time.sleep(duration - elapsed)
                
            data = wf.readframes(chunk_size)
            
        print("\nPlayback finished.")
        wf.close()

if __name__ == "__main__":
    port = get_serial_port()
    if port:
        bridge = AudioBridge(port)
        bridge.listen()
