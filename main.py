from logging import captureWarnings
import subprocess, os, signal, time, argparse
from gpiozero import Button
from gtts import gTTS
from playsound import playsound

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
                print("Entering query mode!")
                self.queryMode_type1()
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
        # up_button = Button(3) # GPIO3 -> Up button
        # down_button = Button(4) # GPIO4 -> Down Button
        self.playVoice("Query mode activaded. Which category do you want?")

        for cat in self.categories:
            self.playVoice(cat)
            if self.query_button.is_pressed:
                selection = cat
                break
            time.sleep(0.5)

        self.playVoice(f"Looking for {selection}.")
        # Start looking for a specific selected item

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
        myobj = gTTS(text=mText, lang=self.lang, slow=False)

        myobj.save("voice.mp3")
        playsound("voice.mp3")
        os.remove("voice.mp3")
        time.sleep(0.5)

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