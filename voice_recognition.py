import speech_recognition as sr
import os
import time
from pydub import AudioSegment
from pydub.playback import play
from gtts import gTTS
from  recording import record_to_file

moc = {"food": ["apple", "banana", "hamburguer", "strawberry"], "automobiles":["car", "motorcycle", "bicycle"]}

class VoiceRecognition():
    def __init__(self, language="en"):
        self.language = language

    def speech_recog(self):
        mic = sr.Recognizer()

        with sr.AudioFile("output.wav") as source:
            audio = mic.record(source)

            try:
                word = mic.recognize_google(audio, language=self.language)
                os.remove("output.wav")
                return word.lower()

            except Exception as e:
                print(e)
                return None
            
    def list_elements(self, list_e):
        list_elem = ""
        for item in list_e:
            list_elem += str(item) +", "
        self.play_voice(list_elements)
        
    def list_categories(self, dictionary):
        list_categories = ""
        for item in list(dictionary.keys()):
            list_categories += str(item) +", "
        self.play_voice(list_categories)


    def play_voice(self, mText):
        tts_audio = gTTS(text=mText, lang=self.language, slow=False)
        tts_audio.save("voice.wav")
        play(AudioSegment.from_file("voice.wav"))
        os.remove("voice.wav")

    def repeat(self, typeof):
        self.play_voice("Could not understand what you said. Which {} do you want?".format(typeof))

    def greetings(self):
        self.play_voice("Query mode activated. Which category do you want?")
    