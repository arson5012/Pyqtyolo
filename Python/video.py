from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from threading import Thread
import speech_recognition as sr
import cv2
import time
from pyfirmata import Arduino, util
import sys
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image
import pytesseract
from gtts import gTTS
import os
import cvlib as cv
from cvlib.object_detection import draw_bbox
import pyautogui
import ctypes



class video(QObject):
    sendImage = pyqtSignal(QImage)

    def __init__(self, widget, size):
        super().__init__()
        self.widget = widget
        self.size = size
        self.sendImage.connect(self.widget.recvImage)

        files = ['haarcascade_frontalface_default.xml',
                 'haarcascade_eye.xml',
                 'haarcascade_eye_tree_eyeglasses.xml']



        self.filters = []
        for i in range(len(files)):
            filter = cv2.CascadeClassifier(files[i])
            self.filters.append(filter)

        self.option = [False for i in range(len(files))]
        self.color = [QColor(255, 0, 0), QColor(255, 128, 0), QColor(255, 255, 0), QColor(0, 255, 0), QColor(0, 0, 255),
                      QColor(0, 0, 128), QColor(128, 0, 128)]

    def setOption(self, option):
        self.option = option


    def startCam(self):
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        except Exception as e:
            print('카메라를 찾을 수 없습니다 : ', e)
        else:
            self.bThread = True
            self.thread = Thread(target=self.threadFunc)
            self.thread.start()

    def stopCam(self):
        self.bThread = False
        bopen = False
        try:
            bopen = self.cap.isOpened()
        except Exception as e:
            print('카메라를 찾을 수 없습니다')
        else:
            self.cap.release()


    def pgup(self):
        while self.jlock:
            ard = Arduino('COM3')
            util.Iterator(ard).start()

            ard.get_pin("d:2:o")

            data_path = 'save_data/'
            onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
            Training_Data, Labels = [], []
            for i, files in enumerate(onlyfiles):
                if not '.jpg' in files:  # 확장자가 jpg가 아닌 경우 무시
                    continue
                image_path = data_path + onlyfiles[i]
                images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                Training_Data.append(np.asarray(images, dtype=np.uint8))
                Labels.append(i)
            if len(Labels) == 0:
                print("배운 정보가 없습니다")
                sys.exit()

            Labels = np.asarray(Labels, dtype=np.int32)
            model = cv2.face.LBPHFaceRecognizer_create()
            model.train(np.asarray(Training_Data), np.asarray(Labels))

            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if cap.isOpened() == False:
                print('카메라를 읽을 수 없습니다')
                return None
            while True:

                ret, frame = cap.read()
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                    detects = face_classifier.detectMultiScale(gray, 1.1, 5)

                    for (x, y, w, h) in detects:
                        global roi
                        roi = frame[y:y + h, x:x + w]
                        roi = cv2.resize(roi, (200, 200))
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        result = model.predict(face)
                        if result[1] < 500:
                            confidence = int(100 * (1 - (result[1]) / 300))
                            display_str = str(confidence) + '% Match Rate'

                        cv2.putText(frame, display_str, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1,
                                    (250, 120, 255), 2)

                        try:
                            cv2.putText(frame, display_str, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1,
                                        (250, 120, 255), 2)
                            if confidence > 74:
                                cv2.putText(frame, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1,
                                            (0, 255, 0), 2)

                                ard.digital[2].write(1)
                                time.sleep(0.5)
                                ard.digital[2].write(0)
                                time.sleep(0.5)
                                ard.digital[2].write(1)
                                time.sleep(0.5)
                                ard.digital[2].write(0)
                                time.sleep(0.5)
                            else:
                                cv2.putText(frame, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1,
                                            (0, 0, 255), 2)

                        except:
                            pass

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    bytesPerLine = ch * w
                    img = QImage(rgb.data, w, h, bytesPerLine, QImage.Format_RGB888)
                    resizedImg = img.scaled(self.size.width(), self.size.height(), Qt.KeepAspectRatio)
                    self.sendImage.emit(resizedImg)

                except cv2.error:
                    ard.exit()
                    cap.release()
                    cv2.destroyAllWindows()
                    pass







    def cvlib2video(self):
        while self.lib:
            ctypes.windll.user32.MessageBoxW(0, "3초 후에 명령을 내리세요", "알림", 64)
            r = sr.Recognizer()
            with sr.Microphone() as source:
                audio = r.listen(source)
                text = r.recognize_google(audio, language='ko')
                pyautogui.alert(r.recognize_google(audio, language='ko'),"STT결과")


                if text=="시작":
                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    if cap.isOpened() == False:
                        print('카메라를 읽을 수 없습니다')
                        return None
                    while True:
                        try:
                            ret, frame = cap.read()
                            bbox, label, conf = cv.detect_common_objects(frame)

                            print(bbox, label, conf)

                            draw_bbox(frame, bbox, label, conf, write_conf=True)

                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            h, w, ch = rgb.shape
                            bytesPerLine = ch * w
                            img = QImage(rgb.data, w, h, bytesPerLine, QImage.Format_RGB888)
                            resizedImg = img.scaled(self.size.width(), self.size.height(), Qt.KeepAspectRatio)
                            self.sendImage.emit(resizedImg)


                        except AttributeError:
                            cap.release()
                            cv2.destroyAllWindows()
                            pass
                else:
                    sr.UnknowValueError: \
                        """print(" ")"""
                    ctypes.windll.user32.MessageBoxW(0, "잘못된 입력입니다 다시 입력 해주세요", "알림", 16)


    def startlib(self):
        try:
            self.cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        except Exception as lr:
            print('카메라를 찾을 수 없습니다 : ', lr)
        else:
            self.lib = True
            self.thread = Thread(target=self.cvlib2video)
            self.thread.start()

    def stoplib(self):
        self.lib = False
        t2 = False
        try:
            t2 = self.cap.isOpened()
        except Exception as lr:
            print('카메라를 찾을 수 없습니다')
        else:
            print("오브젝트 인식을 종료합니다")
            self.cap.release()



    def startlockj(self):
        try:
            self.cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        except Exception as lr:
            print('카메라를 찾을 수 없습니다 : ', lr)
        else:
            self.jlock = True
            self.thread = Thread(target=self.pgup)
            self.thread.start()
    def jatstop(self):
        self.jlock = False
        t2 = False
        try:
            t2 = self.cap.isOpened()
        except Exception as lr:
            print('카메라를 찾을 수 없습니다')
        else:
            self.cap.release()

    def view_word(self):
        while self.bword:
            ctypes.windll.user32.MessageBoxW(0, "3초 후에 명령을 내리세요", "알림", 64)
            r = sr.Recognizer()
            with sr.Microphone() as source:
                audio = r.listen(source)
                text = r.recognize_google(audio, language='ko')
                pyautogui.alert(r.recognize_google(audio, language='ko'), "STT결과")

                if text == "글자 판독기":
                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    if cap.isOpened() == False:
                        print('카메라를 읽을 수 없습니다')
                        return None
                    while True:
                        try:
                            ret, frame = cap.read()
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            for i in range(len(self.filters)):
                                if self.option[i]:
                                    detects = self.filters[i].detectMultiScale(gray, 1.1, 5)
                                    for (x, y, w, h) in detects:
                                        r = self.color[i].red()
                                        g = self.color[i].green()
                                        b = self.color[i].blue()
                                        cv2.rectangle(frame, (x, y), (x + w, y + h), (b, g, r), 2)

                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            h, w, ch = rgb.shape
                            bytesPerLine = ch * w
                            img = QImage(rgb.data, w, h, bytesPerLine, QImage.Format_RGB888)
                            resizedImg = img.scaled(self.size.width(), self.size.height(), Qt.KeepAspectRatio)
                            self.sendImage.emit(resizedImg)

                            pytesseract.pytesseract.tesseract_cmd = r'C:\Users\arson\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
                            filename = "{}.png".format(os.getpid())
                            cv2.imwrite(filename, frame)
                            text = pytesseract.image_to_string(frame, lang=None)
                            os.remove(filename)

                            print(text)

                            f = open('read1.txt', 'wt', encoding="utf-8")
                            f.write(text)

                            f = open('read1.txt', 'rt', encoding="utf-8")
                            mytext = f.read().replace('\n', ' ')

                            try:
                                output = gTTS(text=mytext, lang='en')
                                output.save("save1.mp3")
                                os.system('save1.mp3')
                            except:
                                print("글자가 없거나 읽을 수 없습니다")
                        except cv2.error:
                            cap.release()
                            cv2.destroyAllWindows()
                            pass
                else:
                    sr.UnknowValueError: \
                        """print(" ")"""
                    ctypes.windll.user32.MessageBoxW(0, "잘못된 입력입니다 다시 입력 해주세요", "알림", 16)

    def startword(self):
        try:
            self.cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        except Exception as zk:
            print('카메라를 찾을 수 없습니다 : ', zk)
        else:
            self.bword = True
            self.thread = Thread(target=self.view_word)
            self.thread.start()

    def stopword(self):
        self.bword = False
        bstop = False
        try:
            bstop = self.cap.isOpened()
        except Exception as zk:
            print('카메라를 찾을 수 없습니다')
        else:
            self.cap.release()



    def threadFunc(self):
        while self.bThread:
            ok, frame = self.cap.read()
            if ok:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for i in range(len(self.filters)):
                    if self.option[i]:
                        detects = self.filters[i].detectMultiScale(gray, 1.1, 5)
                        for (x, y, w, h) in detects:
                            r = self.color[i].red()
                            g = self.color[i].green()
                            b = self.color[i].blue()
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (b, g, r), 2)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytesPerLine = ch * w
                img = QImage(rgb.data, w, h, bytesPerLine, QImage.Format_RGB888)
                resizedImg = img.scaled(self.size.width(), self.size.height(), Qt.KeepAspectRatio)
                self.sendImage.emit(resizedImg)
            else:
                print('카메라를 읽을 수 없습니다')




    def close1(self):
        try:
            self.cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        except Exception as lr:
            pass
        else:
            self.C = True
            self.thread = Thread(target=self.close)
            self.thread.start()