import os
import pathlib
import threading
import winsound
from datetime import datetime

import cv2
import geocoder
from playsound import playsound

# import winsound


path = pathlib.Path('./resources/sounds/its_basic_alert.wav')
wavFile = os.path.abspath(path)
font = cv2.FONT_HERSHEY_COMPLEX

# WINDOWS ONLY
def alert_user(reason):
    threading.Thread(target=playsound, args=(wavFile,), daemon=True).start()
    #playsound(wavFile,False)
    # winsound.PlaySound(wavFile, winsound.SND_ASYNC | winsound.SND_ALIAS)
    # TODO : add warn to db, Firebase ?
    # date, position, reason...
    position = geocoder.ip('me')  # .latlng
    now = datetime.now().strftime("%d/%m/%Y|%H:%M:%S:")
    print("ALERT USER :",now, " ", reason, "(", position, ")")

def alert_user_linux(reason):
    threading.Thread(target=playsound, args=(wavFile,), daemon=True).start()
    position = geocoder.ip('me')  # .latlng
    now = datetime.now().strftime("%d/%m/%Y|%H:%M:%S:")
    print("ALERT USER :", now, " ", reason, "(", position, ")")


def write_alert(img,txt,x,y):
    cv2.putText(img, txt, (x, y), font, 1, (0))


if __name__ == '__main__':
    alert_user("")
    #alert_user_linux("aled")
