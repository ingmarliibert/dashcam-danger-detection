import os
import pathlib

from playsound import playsound

path = pathlib.Path('./resources/sounds/its_basic_alert.wav')
wavFile = os.path.abspath(path)


def alert_user(reason):
    playsound(wavFile)
    # TODO : add warn to db, Firebase ?
    # date, position, reason...
    # position = geocoder.ip('me')  # .latlng

    # now = datetime.now().strftime("%d/%m/%Y|%H:%M:%S:")
    # print(now, " ", reason, "(", position, ")")


if __name__ == '__main__':
    alert_user("")
