import subprocess, os, signal, time, argparse
from gpiozero import Button

def main(MODEL_NAME, RESOLUTION, USE_EDGETPU, CATEGORIES):
    """Function that orchestrates the whole system"""
    # Start running safari mode
    # safariProcess = safari()

    # Conect button on GPIO2 and Ground
    # Watch out for connenctions in 'pin_layout.svg'
    query_button = Button(2)
    while (True):
        if query_button.is_pressed:
            # Enter Query Mode
            print("Entering query mode!")
            up_button = Button(3) # GPIO3 -> Up button
            down_button = Button(4) # GPIO4 -> Down Button
            # Kills safari mode
            # os.killpg(os.getpgid(safariProcess.pid), signal.SIGTERM)

            # Starts tts

        else:
            print("Button was not pressed...")
        time.sleep(1)
        
def getAllCategories(dir):
    """Needs testing!"""
    """Function that get all available categories from model '.name' file"""
    for root, dir, files in os.walk(dir):
        for f in files:
            if ".name" in f:
                filename = f
                break
    cat = []
    f = open(filename, "r")
    for line in f.readlines():
        cat.append(line)
    return cat

def safari(modeldir, resolution, edgetpu):
    # Needs testing!
    p = subprocess.Popen(f"sudo python3 TFLite_detection_webcam --modeldir=Sample_TFLite_model --edgetpu", stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
    # output, err = p.communicate()
    # output.decode().split()[0]
    return p
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in', 
                        default="Sample_TFLite_model/")
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='1280x720')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                        action='store_true')
    args = parser.parse_args()

    MODEL_NAME = args.modeldir
    RESOLUTION = args.resolution
    USE_EDGETPU = args.edgetpu
    categories = getAllCategories(MODEL_NAME)
    main(MODEL_NAME, RESOLUTION, USE_EDGETPU, categories)