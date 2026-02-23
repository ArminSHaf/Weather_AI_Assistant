import os
import sys
import asyncio
import threading
import time
import speech_recognition as sr
import pyttsx3
from dotenv import load_dotenv

# ADK & Google AI
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types
from google.adk.models.google_llm import Gemini

# Local Modules
import prediction

# Add parent dir to find stream_audio.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import stream_audio

# --- CONFIGURATION ---
load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    print("CRITICAL: GOOGLE_API_KEY not found.")
    sys.exit(1)

# --- TOOLS ---
def transcribe(path: str) -> str:
    r = sr.Recognizer()
    with sr.AudioFile(path) as source:
        audio = r.record(source)
    return r.recognize_google(audio, language="en-US")

def text_to_speech(text: str, out_path: str):
    try:
        engine = pyttsx3.init()
        # Try to find a female voice (Zira)
        voices = engine.getProperty('voices')
        for v in voices:
            if "zira" in v.name.lower() or "female" in v.name.lower():
                engine.setProperty('voice', v.id)
                break
        engine.save_to_file(text, out_path)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")

def weather_tool(city_name: str) -> dict:
    print(f"DEBUG: calling weather_tool for {city_name}")
    return prediction.predict_weather(city_name)

# --- AI AGENT ---
retry_config = types.HttpRetryOptions(
    attempts=5, exp_base=2, initial_delay=1, http_status_codes=[429, 500, 503]
)

weather_agent = Agent(
    name="weather_predictor",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    You are a professional Weather assistant.
    IMPORTANT: You cannot know the weather yourself. You MUST use the 'weather_tool'.
    1. Identify the city name. If not found -> default to 'Bucharest'.
    2. Call 'weather_tool(city_name)'.
    3. Summarize the result in MAX 2 sentences (Conditions + Advice).
    4. Start with: "Great, here are the results for [City Name]."
    """,
    tools=[weather_tool]
)

# --- MAIN LOOP ---
async def main():
    print("--- 🌦️ Weather AI Assistant 🌦️ ---")
    
    # 1. Start Audio Bridge (ESP32 <-> PC)
    print("Initializing Audio Bridge...")
    port = stream_audio.get_serial_port()
    if not port:
        print("No serial port found. Exiting.")
        return

    bridge = stream_audio.AudioBridge(port)
    threading.Thread(target=bridge.listen, daemon=True).start()
    
    print("\nREADY! Press the button on your ESP32 to speak.")

    # 2. Setup AI Runner
    APP_NAME = "weather_app"
    USER_ID = "user"
    runner = InMemoryRunner(agent=weather_agent, app_name=APP_NAME)
    
    # 3. File Paths
    audio_dir = "audio_folder"
    audio_path = os.path.join(audio_dir, "audio.wav")
    reply_path = os.path.join(audio_dir, "reply.wav")

    # Cleanup old reply
    if os.path.exists(reply_path):
        try: os.remove(reply_path)
        except: pass

    last_mtime = 0
    if os.path.exists(audio_path):
        last_mtime = os.path.getmtime(audio_path)

    # 4. Watch Loop
    while True:
        try:
            if os.path.exists(audio_path):
                mtime = os.path.getmtime(audio_path)
                if mtime > last_mtime:
                    print("\n🎤 New audio detected! Processing...")
                    last_mtime = mtime
                    await asyncio.sleep(0.5) 
                    
                    # A. Transcribe
                    try:
                        user_text = transcribe(audio_path)
                        print(f"User said: '{user_text}'")
                    except Exception as e:
                        print(f"Transcription Error (ignoring): {e}")
                        continue

                    # B. AI Response
                    content = types.Content(role="user", parts=[types.Part.from_text(text=user_text)])
                    
                    # --- CRITICAL FIX: Create FRESH session for every request to avoid 'Session not found' ---
                    print("Creating fresh session...")
                    try:
                        session = await runner.session_service.create_session(user_id=USER_ID, app_name=APP_NAME)
                        
                        async for event in runner.run_async(session_id=session.id, user_id=USER_ID, new_message=content):
                             if event.content and event.content.parts:
                                for part in event.content.parts:
                                    if part.text:
                                        print(f"Agent: {part.text}")
                                        text_to_speech(part.text, reply_path)
                                        print("🔊 Reply sent to ESP32.")
                    except Exception as e:
                         print(f"AI/Session Error: {e}")
                         import traceback
                         traceback.print_exc()

            await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            print("\nStopping...")
            break
        except Exception as e:
            print(f"Loop Error: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())

# import os
# import sys
# import asyncio
# from dotenv import load_dotenv

# # ADK Imports
# from google.adk.agents import Agent
# from google.adk.runners import InMemoryRunner
# from google.genai import types
# from google.adk.models.google_llm import Gemini

# # prediction.py
# import prediction

# # speech rec
# import speech_recognition as sr
# from gtts import gTTS


# def transcribe(path: str) -> str:
#     r = sr.Recognizer()
#     with sr.AudioFile(path) as source:
#         audio = r.record(source)

#     return r.recognize_google(audio,  language="en-US")


# def text_to_speech(text: str, out_path: str):
#     tts = gTTS(text=text, lang="en")
#     tts.save(out_path)


# # Load environment variables
# load_dotenv()

# if not os.environ.get("GOOGLE_API_KEY"):
#     print("CRITICAL: GOOGLE_API_KEY not found. Please check your.env file or create one.")
#     sys.exit(1)


# def weather_tool(city_name: str) -> dict:
#     """
#     Get the current weather and forecast for a specific city.

#     Args:
#         city_name: The name of the city to look up.
#     """
#     return prediction.predict_weather(city_name)


# retry_config = types.HttpRetryOptions(
#     attempts=5,  # Maximum retry attempts
#     exp_base=7,  # Delay multiplier
#     initial_delay=1,  # Initial delay before first retry (in seconds)
#     http_status_codes=[429, 500, 503, 504]  # Retry on these HTTP errors
# )


# weather_agent = Agent(
#     name="weather_predictor",
#     model=Gemini(
#         model="gemini-2.5-flash-lite",
#         retry_options=retry_config
#     ),
#     instruction="""
#     You are a professional Weather assistant.
#     1. Identify the city name from the user's request. if not identified -> default city name : Bucharest
#     2. Call the 'weather_tool' passing the city name.
#     3. Analyze the returned data (current_observation vs forecast). Give human friendly - short sentences. **MAX 2 sentences**
#     4. Announce the weather condition and how it will change, say  in few next hours we will reach minus 1 degree with the wind ranging from (x to y)
#     5. Give specific advice on what to wear and items to take.
#     6. start with : Great, here are some results I found for ... ( the city name ) 
#     """,
#     tools=[weather_tool]
# )

# # --- 3. Main Interaction Loop ---


# async def main():
#     print("--- 🌦️ weather ai agent  🌦️ ---")
#     print("Hi there! this is the Weather AI assistant. Tell me what I can help with?")

#     APP_NAME = "weather_app"
#     USER_ID = "local_user"
#     runner = InMemoryRunner(agent=weather_agent, app_name=APP_NAME)

#     # Create the session
#     session = await runner.session_service.create_session(
#         user_id=USER_ID,
#         app_name=APP_NAME
#     )

#     # Save the ID as a string for use in the loop
#     current_session_id = session.id
#     print(f"DEBUG: Session Created with ID: {current_session_id}")

#     audio_dir = "audio_folder"
#     audio_path = os.path.join(audio_dir, "audio.wav")
#     reply_path = os.path.join(audio_dir, "reply.wav")
#     while True:
#         try:
#             user_input = input("\nUser: ").strip()
#             if user_input.lower() in ["quit", "exit"]:
#                 print("Exiting...")
#                 break

#             print("... Got it ! thinking...")
#             try:
#                 user_input = transcribe(audio_path)
#             except Exception as e:
#                 print("transcription failed", e)
#                 continue

#             # Prepare the content object (Multimodal structure)
#             user_content = types.Content(
#                 role="user",
#                 parts=[types.Part.from_text(text=user_input)]
#             )

#             # Use run_async (Asynchronous Generator)
#             async for event in runner.run_async(
#                 session_id=current_session_id,
#                 user_id=USER_ID,
#                 new_message=user_content
#             ):
#                 if event.content and event.content.parts:
#                     for part in event.content.parts:
#                         if part.text:
#                             text_to_speech(part.text, reply_path)
#                             print(f"Agent: {part.text}")

#         except Exception as e:
#             print(f"Error occurred: {e}")
#             import traceback
#             traceback.print_exc()

# if __name__ == "__main__":
#     asyncio.run(main())