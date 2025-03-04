import os
import speech_recognition as sr
import streamlit as st
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SpeechHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.audio_temp_file = os.getenv("AUDIO_TEMP_FILE", "temp_audio.mp3")
        self.timeout = int(os.getenv("SPEECH_RECOGNITION_TIMEOUT", "50"))
        self.phrase_time_limit = int(os.getenv("SPEECH_RECOGNITION_PHRASE_TIME_LIMIT", "100"))
        self.default_language = os.getenv("DEFAULT_LANGUAGE", "en")
        self.is_listening = False
        self.is_processing = False
        self.audio_data = []
        self.energy_levels = []
        self.listen_thread = None
        self.complete_transcript = ""
        self.status = "idle"  # idle, listening, transcribing, processing

    def speak_text(self, text, language=None):
        """Convert text to speech and play it"""
        if not text:
            return False
        
        try:
            self.status = "speaking"
            lang = language or self.default_language
            tts = gTTS(text=text, lang=lang)
            tts.save(self.audio_temp_file)
            audio = AudioSegment.from_mp3(self.audio_temp_file)
            play(audio)
            
            # Clean up
            if os.path.exists(self.audio_temp_file):
                os.remove(self.audio_temp_file)
            
            self.status = "idle"
            return True
        except Exception as e:
            st.error(f"Error in text-to-speech: {str(e)}")
            self.status = "idle"
            return False

    def start_listening(self, callback=None):
        """Start listening in a separate thread"""
        if self.is_listening:
            st.warning("Already listening")
            return False
        
        try:
            # Test microphone access before starting thread
            with sr.Microphone() as source:
                st.info("Microphone initialized successfully")
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            self.is_listening = True
            self.status = "listening"
            self.audio_data = []
            self.energy_levels = []
            self.complete_transcript = ""
            
            # Start listening in a separate thread
            self.listen_thread = threading.Thread(
                target=self._listen_continuously, 
                args=(callback,)
            )
            self.listen_thread.daemon = True
            self.listen_thread.start()
            st.success("Started listening thread")
            return True
        except Exception as e:
            st.error(f"Error initializing microphone: {str(e)}")
            self.is_listening = False
            self.status = "idle"
            return False

    def stop_listening(self):
        """Stop the listening process"""
        self.is_listening = False
        self.status = "transcribing"
        
        # Wait for the thread to finish if it's running
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=2.0)
        
        # Process any remaining audio data
        self._process_audio_data()
        self.status = "idle"
        return self.complete_transcript

    def _listen_continuously(self, callback=None):
        """Continuously listen for audio input"""
        try:
            st.info("Starting continuous listening")
            with sr.Microphone() as source:
                # Adjust for ambient noise
                st.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                st.info("Ambient noise adjustment complete")
                
                while self.is_listening:
                    try:
                        # Record audio
                        st.info("Listening for speech...")
                        audio = self.recognizer.listen(
                            source, 
                            timeout=self.timeout, 
                            phrase_time_limit=self.phrase_time_limit
                        )
                        st.info("Captured audio segment")
                        self.audio_data.append(audio)
                        
                        # Get audio energy levels for visualization
                        frame_data = np.frombuffer(audio.frame_data, dtype=np.int16)
                        energy = np.sqrt(np.mean(frame_data**2))
                        self.energy_levels.append(energy)
                        
                        # Process the audio in real-time
                        self.status = "transcribing"
                        st.info("Transcribing audio...")
                        text = self.recognizer.recognize_google(audio)
                        st.success(f"Transcribed: {text}")
                        self.complete_transcript += " " + text
                        self.status = "listening"
                        
                        # Call the callback function if provided
                        if callback and callable(callback):
                            callback(text, False)  # False indicates it's not the final transcript
                    
                    except sr.WaitTimeoutError:
                        # Timeout occurred, continue listening
                        st.warning("Listening timeout, continuing...")
                        pass
                    except sr.UnknownValueError:
                        # Speech was unintelligible
                        st.warning("Speech unintelligible, continuing...")
                        pass
                    except Exception as e:
                        st.error(f"Error during listening: {str(e)}")
                        break
        
        except Exception as e:
            st.error(f"Error initializing microphone: {str(e)}")
        
        finally:
            st.info("Exiting listening thread")
            self.is_listening = False

    def _process_audio_data(self):
        """Process all collected audio data to get a complete transcript"""
        if not self.audio_data:
            return ""
        
        combined_transcript = self.complete_transcript
        
        # Process any remaining audio that hasn't been transcribed yet
        for audio in self.audio_data:
            try:
                text = self.recognizer.recognize_google(audio)
                if text and text not in combined_transcript:
                    combined_transcript += " " + text
            except:
                pass
        
        self.complete_transcript = combined_transcript.strip()
        return self.complete_transcript

    def get_audio_visualization(self):
        """Generate a visualization of audio energy levels"""
        if not self.energy_levels:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot(self.energy_levels)
        ax.set_title("Audio Energy Levels")
        ax.set_xlabel("Time")
        ax.set_ylabel("Energy")
        ax.grid(True)
        
        return fig

    def get_status(self):
        """Return the current status of the speech handler"""
        return self.status
