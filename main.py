from logging import captureWarnings
import subprocess, os, signal, time, argparse

# GPIO - Pi Buttons
from gpiozero import Button

# Audio
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

# Threads
import threading

# TFLite detection
from TFLite_detection_webcam import initialize_detector, safari_mode, query_mode


class VMobi:
    """Class that represents the system as a whole"""

    def __init__(self, args, lang = "en"):
        self.args = args # Saving arguments on a variable
        self.MODEL_DIR = args.modeldir # Directory of the .tflite file and names file
        self.RESOLUTION = args.resolution # Camera resolution as in pixels 
        self.USE_EDGETPU = args.edgetpu # Flag to use the google coral tpu
        self.lang = lang # Language used on tts speech voice

        self.main() # Runs on the raspberry with buttons on the GPIO
        # self.test() # Function to test tts on PC

    def test(self):
        categories = self.get_all_categories()
        print(categories)
        time.sleep(1)
        print("Now playing the audio")
        play_voice("Query mode activaded. Which category do you want?", self.lang)
        for cat in categories:
            play_voice(cat, self.lang)

    def main(self):
        """Main function that orchestrates the product"""
        print("On main function!")

        # Get a list of the categories as strings
        self.categories = self.get_all_categories()
        print(f"Got all categories: {self.categories}")

        # Conect button on GPIO2 and Ground
        # Watch out for connenctions in 'pin_layout.svg'
        self.query_button = Button(2)

        # Running the safari mode to run on the background
        # thread_safari_mode = threading.Thread(target=initialize_detector, args=(self.args,))
        # thread_safari_mode.start()
        detector_args = initialize_detector(self.args)
            
        while (True):
            s = safari_mode(detector_args)
            if s > 0:
                # Enter Query Mode
                query_cat = self.query_mode_selection() # Get the category with the GPIO buttons
                query_mode(detector_args, query_cat)
                continue

    def query_mode_selection(self):
        """[Type 1] Query mode that functions only with buttons"""
        up_button = Button(18) # GPIO18 -> Up button
        down_button = Button(23) # GPIO23 -> Down Button
        print("Entering query mode only with buttons. (Type 1)")
        play_voice("Query mode activaded. Which category do you want?", self.lang)
        selection = None

        index = 0
        while True:
            if (self.query_button.is_pressed or up_button.is_pressed or down_button.is_pressed):
                if up_button.is_pressed:
                    if (index + 1 >= len(self.categories)):
                        index = 0
                        continue
                    print("Up Button was pressed!")
                    index += 1
                if down_button.is_pressed:
                    if (index - 1 < 0):
                        index = len(self.categories) - 1
                        continue
                    print("Down Button was pressed!")
                    index -= 1
                if self.query_button.is_pressed:
                    # User choosed the category self.categories[index]
                    selection = self.categories[index]
                    print("Query Button was pressed!")
                    break
                play_voice(self.categories[index], self.lang)
        
        play_voice(f"You chose the category: {selection}", self.lang)
        return selection
            

    def query_mode_type2(self):
        """Query  mode that uses voice recognition and only the query button"""
        print("Entering query mode with voice recognition. (Type 2)")
        play_voice("Query mode activaded. Which category do you want?", self.lang)
        ##########################################################################
        #                               Pseudo-code                              #
        # ******************************** TODO ******************************** #
        # Start voice recognition                                                #
        # while (True):                                                          #
        #     if user says something:                                            #
        #         if it is a known category:                                     #
        #             play_voice(f"Looking for {selection}", self.lang)          #
        #             break                                                      #
        #         elif contains "list":                                          #
        #             for cat in self.categories:                                #
        #                 play_voice(cat, self.lang)                             #
        #                 if self.query_button.is_pressed:                       #
        #                     break                                              #
        #         else:                                                          #
        #             play_voice(f"Unknown category {selection}.", self.lang)    #
        #             play_voice(f"Please repeat the category name.", self.lang) #
        #             continue                                                   #
        # Start looking for the specific object                                  #
        ##########################################################################


    def get_all_categories(self):
        """Function that get all available categories from model '.name' file"""
        for root, dir, files in os.walk(self.MODEL_DIR):
            for f in files:
                if "labelmap.txt" in f:
                    filename = f
                    break
        cat = []

        f = open(self.MODEL_DIR + filename, "r")
        for line in f.readlines():
            if "?" in line:
                continue
            cat.append(line.replace("\n", ""))
        return cat

def play_voice(mText, lang="en"):
        """Function used to play the string 'mText' in audio using tts"""
        print(f"[play_voice] now playing: '{mText}'")
        tts_audio = gTTS(text=mText, lang=lang, slow=False)

        tts_audio.save("voice.wav")
        play(AudioSegment.from_file("voice.wav"))
        os.remove("voice.wav")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in', 
                        default="Sample_TFLite_model/")
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                        default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                        default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                        default=0.5)
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='1280x720')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                        action='store_true')
    parser.add_argument('--safari', help='Start Safari Mode', action='store_true')
    parser.add_argument('--query', help='Start Query Mode', default='?')

    args = parser.parse_args()

    helper = VMobi(args)