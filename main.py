import subprocess, os, signal, time
from gpiozero import Button

def main():
    """Function that orchestrates the whole system"""
    # Start running safari mode
    # safari()

    # Conect button on GPIO2 and Ground
    but = Button(2)
    while (True):
        if but.is_pressed:
            # Enter Query Mode
            print("Entering query mode!")
        else:
            print("Button was not pressed...")
        time.sleep(1)
        


def safari():
    p = subprocess.Popen(f"sudo python3 TFLite_detection_webcam --modeldir=Sample_TFLite_model --edgetpu", stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
    
    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    # output, err = p.communicate()
    # output.decode().split()[0]
    

if __name__ == '__main__':
    main()