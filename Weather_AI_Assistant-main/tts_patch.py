
def text_to_speech(text: str, out_path: str):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    # Try to find a female voice
    for voice in voices:
        if "zira" in voice.name.lower() or "female" in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
    # Fallback to index 1 if available (usually Zira on Windows)
    else:
        if len(voices) > 1:
            engine.setProperty('voice', voices[1].id)
            
    engine.save_to_file(text, out_path)
    engine.runAndWait()
