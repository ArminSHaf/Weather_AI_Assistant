import os
import sys
import asyncio
from dotenv import load_dotenv

# ADK Imports
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types
from google.adk.models.google_llm import Gemini

# prediction.py
import prediction

# speech rec
import speech_recognition as sr
from gtts import gTTS


def transcribe(path: str) -> str:
    r = sr.Recognizer()
    with sr.AudioFile(path) as source:
        audio = r.record(source)

    return r.recognize_google(audio,  language="en-US")


def text_to_speech(text: str, out_path: str):
    tts = gTTS(text=text, lang="en")
    tts.save(out_path)


# Load environment variables
load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    print("CRITICAL: GOOGLE_API_KEY not found. Please check your.env file or create one.")
    sys.exit(1)


def weather_tool(city_name: str) -> dict:
    """
    Get the current weather and forecast for a specific city.

    Args:
        city_name: The name of the city to look up.
    """
    return prediction.predict_weather(city_name)


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,  # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504]  # Retry on these HTTP errors
)


weather_agent = Agent(
    name="weather_predictor",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    You are a professional Weather assistant.
    1. Identify the city name from the user's request. if not identified -> default city name : Bucharest
    2. Call the 'weather_tool' passing the city name.
    3. Analyze the returned data (current_observation vs forecast). Give human friendly - short sentences. **MAX 2 sentences**
    4. Announce the weather condition and how it will change, say  in few next hours we will reach minus 1 degree with the wind ranging from (x to y)
    5. Give specific advice on what to wear and items to take.
    6. start with : Great, here are some results I found for ... ( the city name ) 
    """,
    tools=[weather_tool]
)

# --- 3. Main Interaction Loop ---


async def main():
    print("--- 🌦️ weather ai agent  🌦️ ---")
    print("Hi there! this is the Weather AI assistant. Tell me what I can help with?")

    APP_NAME = "weather_app"
    USER_ID = "local_user"
    runner = InMemoryRunner(agent=weather_agent, app_name=APP_NAME)

    # Create the session
    session = await runner.session_service.create_session(
        user_id=USER_ID,
        app_name=APP_NAME
    )

    # Save the ID as a string for use in the loop
    current_session_id = session.id
    print(f"DEBUG: Session Created with ID: {current_session_id}")

    audio_dir = "audio_folder"
    audio_path = os.path.join(audio_dir, "audio.wav")
    reply_path = os.path.join(audio_dir, "reply.wav")
    while True:
        try:
            user_input = input("\nUser: ").strip()
            if user_input.lower() in ["quit", "exit"]:
                print("Exiting...")
                break

            print("... Got it ! thinking...")
            try:
                user_input = transcribe(audio_path)
            except Exception as e:
                print("transcription failed", e)
                continue

            # Prepare the content object (Multimodal structure)
            user_content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)]
            )

            # Use run_async (Asynchronous Generator)
            async for event in runner.run_async(
                session_id=current_session_id,
                user_id=USER_ID,
                new_message=user_content
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            text_to_speech(part.text, reply_path)
                            print(f"Agent: {part.text}")

        except Exception as e:
            print(f"Error occurred: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
