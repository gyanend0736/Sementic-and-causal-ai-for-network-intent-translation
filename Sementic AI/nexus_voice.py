import speech_recognition as sr
import pyttsx3
import requests

# --- 1. Setup the AI Voice (Offline, Zero Latency) ---
engine = pyttsx3.init()
engine.setProperty('rate', 175) 

def speak(text):
    print(f"\n[AI]: {text}")
    engine.say(text)
    engine.runAndWait()

# --- 2. Setup the Ear (With Anti-Spam Settings) ---
recognizer = sr.Recognizer()
recognizer.energy_threshold = 4000  # Require a loud, clear voice
recognizer.dynamic_energy_threshold = False  # Ignore fan noise
recognizer.pause_threshold = 0.8  # Ignore static pops

WAKE_WORD = "ram"
FLASK_URL = "http://127.0.0.1:5000/translate"

# --- 3. The Flask Bridge (The Missing Function!) ---
def send_to_causal_engine(command):
    """Sends the transcribed text to your existing Flask backend"""
    try:
        print("here")
        response = requests.post(FLASK_URL, json={"text": command})
        data = response.json()
        print("here")
        # Check if the automated admin resolved a conflict
        if "causal_analysis" in data and data["causal_analysis"].get("resolution_found"):
            limit = data["causal_analysis"]["new_bandwidth_limit_mbps"]
            target = data["causal_analysis"]["target_service"]
            speak(f"Action deployed. Throttled {target} down to {limit} megabits per second. Your application is prioritized.")
            
        # Check if it was a safe proactive intent
        elif "causal_analysis" in data and data["causal_analysis"].get("is_safe"):
            speak("Network intent is mathematically safe. Deployment complete.")
            
        # If it failed or crashed the switch
        else:
            speak("Warning. Conflict predicted or intent unsafe. Action blocked.")
            
    except requests.exceptions.ConnectionError:
        print("[!] Error: Could not connect to Flask. Is app.py running?")
        speak("System offline. Cannot reach the network orchestrator.")

# --- 4. The Listening Engine ---
def listen_and_process():
    """Continuous listening loop with debugging"""
    with sr.Microphone() as source:
        print("\n[System] Calibrating to background noise... (Please stay silent for 2 seconds)")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print(f"[System] Online. Say '{WAKE_WORD}' followed by your command.")
        speak("Network orchestrator online.")
        
        while True:
            print("\n[*] Microphones open. Waiting for you to speak...")
            try:
                # Listen continuously
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                print("[*] Audio captured! Sending to Google for translation...")
                
                # Transcribe
                text = recognizer.recognize_google(audio).lower().strip()
                print(f"[RAW AUDIO DETECTED]: '{text}'")
                
                # Check for Wake Word
                if WAKE_WORD in text:
                    print(f">> WAKE WORD '{WAKE_WORD.upper()}' ACCEPTED!")
                    
                    # Strip the wake word out to get the raw command
                    command = text.replace(WAKE_WORD, "").strip()
                    
                    if command:
                        speak("Processing intent.")
                        send_to_causal_engine(command)
                    else:
                        speak("Awaiting network intent.")
                        
            except sr.WaitTimeoutError:
                pass 
            except sr.UnknownValueError:
                print("[*] (Heard some noise, but couldn't understand any words)")
            except sr.RequestError as e:
                print(f"\n[!] Internet Connection Error: {e}")
            except Exception as e:
                print(f"\n[!] Hardware Error: {e}")

if __name__ == "__main__":
    listen_and_process()