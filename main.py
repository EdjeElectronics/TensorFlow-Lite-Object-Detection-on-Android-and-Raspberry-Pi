from logging import captureWarnings
import subprocess, os, signal, time, argparse

# GPIO - Pi Buttons
from gpiozero import Button

# Audio
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play


class VMobi:
    """Class that represents the system as a whole"""

    def __init__(self, args, lang = "en"):
        self.MODEL_DIR = args.modeldir # Directory of the .tflite file and names file
        self.RESOLUTION = args.resolution # Camera resolution as in pixels 
        self.USE_EDGETPU = args.edgetpu # Flag to use the google coral tpu
        self.lang = lang # Language used on tts speech voice

        self.main() # Runs on the raspberry with buttons on the GPIO
        # self.test() # Function to test tts on PC

    def test(self):
        categories = self.getAllCategories()
        print(categories)
        time.sleep(1)
        print("Now playing the audio")
        self.playVoice("Query mode activaded. Which category do you want?")
        for cat in categories:
            self.playVoice(cat)

    def main(self):
        """Main function that orchestrates the product"""
        print("On main function!")

        # Get a list of the categories as strings
        self.categories = self.getAllCategories()
        print(f"Got all categories: {self.categories}")

        # Running the safari mode to run on the background
        # safari_proccess = self.safari()

        # Conect button on GPIO2 and Ground
        # Watch out for connenctions in 'pin_layout.svg'
        self.query_button = Button(2)
        while (True):
            if self.query_button.is_pressed:
                # Enter Query Mode
                self.queryMode_type1()
                # self.queryMode_type2()
                # Kills safari mode
                # os.killpg(os.getpgid(safariProcess.pid), signal.SIGTERM)
    #     time.sleep(1)

    def safari(self):
        """Function that start the MobileNet SSD v2 detection -- Safari Mode"""
        # Needs testing!
        if (self.USE_EDGETPU):
            p = subprocess.Popen(f"sudo python3 TFLite_detection_webcam --modeldir={self.MODEL_DIR} --resolution={self.RESOLUTION} --edgetpu", stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        else:
            p = subprocess.Popen(f"sudo python3 TFLite_detection_webcam --modeldir={self.MODEL_DIR} --resolution={self.RESOLUTION}", stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        
        # If get the output is needed, do as below:
        # output, err = p.communicate()
        # output.decode().split()[0] # This will, be a formatted string of the output
        return p

    def queryMode_type1(self):
        """Query mode that functions only with buttons"""
        up_button = Button(18) # GPIO18 -> Up button
        down_button = Button(23) # GPIO23 -> Down Button
        print("Entering query mode only with buttons. (Type 1)")
        self.playVoice("Query mode activaded. Which category do you want?")
        self.playVoice("Use up and doown buttons to navigate and query button to select the category you want.")
        selection = None
        ############
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
                self.playVoice(self.categories[index])
        
        self.playVoice(f"You choosed the category: {selection}")
        return selection
            # if up is pressed:
            #    index += 1
            #    continue
            # if down is pressed:
            #    index -= 1
            #    continue
            # if query is pressed:
            #    Choose category
            # 
            # if query is hold:
            #    Cancel
            #    search = False
            #    break
        #########
        # while search:
        #     for cat in self.categories:
        #         self.playVoice(cat)
        #         t0 = time.time()
        #         while (time.time() - t0 < 0.25):
        #             if self.query_button.is_pressed:
        #                 selection = cat
        #                 search = False
        #                 break

        #     if selection == None:
        #         self.playVoice(f"You want to listen again to the categories?")
        #         # self.query_button.held_time = 3 # Held time set for 3 seconds
        #         self.query_button.wait_for_press() # Wait until the query button is pressed
                
        #         if (self.query_button.is_held): # if it is held
        #             print("Held query button, getting back to safari mode")
        #             self.playVoice("Back to safari mode")
        #             return

        #         elif (self.query_button.is_pressed): # if it is only pressed
        #             print("Readig list again...")
        #             break

        #     else:
        #         break
        # self.playVoice(f"Looking for {selection}.")
        # print(f"You selected {selection}")
        # Start looking for a specific selected item

    def queryMode_type2(self):
        """Query  mode that uses voice recognition and only the query button"""
        print("Entering query mode with voice recognition. (Type 2)")
        self.playVoice("Query mode activaded. Which category do you want?")
        ###################################################################
        #                          Pseudo-code                            #
        # *************************** TODO ****************************** #
        # Start voice recognition                                         #
        # while (True):                                                   #
        #     if user says something:                                     #
        #         if it is a known category:                              #
        #             self.playVoice(f"Looking for {selection}")          #
        #             break                                               #
        #         elif contains "list":                                   #
        #             for cat in self.categories:                         #
        #                 self.playVoice(cat)                             #
        #                 if self.query_button.is_pressed:                #
        #                     break                                       #
        #         else:                                                   #
        #             self.playVoice(f"Unknown category {selection}.")    #
        #             self.playVoice(f"Please repeat the category name.") #
        #             continue                                            #
        # Start looking for the specific object                           #
        ###################################################################


    def getAllCategories(self):
        """Function that get all available categories from model '.name' file"""
        for root, dir, files in os.walk(self.MODEL_DIR):
            for f in files:
                if ".name" in f:
                    filename = f
                    break
        cat = []

        f = open(self.MODEL_DIR + filename, "r")
        for line in f.readlines():
            cat.append(line.replace("\n", ""))
        return cat

    def playVoice(self, mText):
        """Function used to play the string 'mText' in audio using tts"""
        print(f"[playVoice] now playing: '{mText}'")
        tts_audio = gTTS(text=mText, lang=self.lang, slow=False)

        tts_audio.save("voice.wav")
        play(AudioSegment.from_file("voice.wav"))
        os.remove("voice.wav")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in', 
                        default="Sample_TFLite_model/")
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='1280x720')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                        action='store_true')
    args = parser.parse_args()

    helper = VMobi(args)