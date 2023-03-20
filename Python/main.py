from PyQt5.QtWidgets import *
from video import *
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from threading import Thread
import speech_recognition as sr
import time
from pyfirmata import Arduino, util
from PIL import Image
import pytesseract
from gtts import gTTS
import sys
 
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
 
class CWidget(QWidget):
 
    def __init__(self):
        super().__init__()
        size = QSize(600,500)
        self.initUI(size)
        self.video = video(self, QSize(self.frm.width(), self.frm.height()))

    def initUI(self, size):

        vbox = QVBoxLayout()
        self.btn = QPushButton('실시간 인식', self)
        self.btn.setCheckable(True)
        self.btn.clicked.connect(self.onoffCam)
        vbox.addWidget(self.btn)

        self.btn3 = QPushButton('잠금 해제', self)
        self.btn3.setCheckable(True)
        self.btn3.clicked.connect(self.locking)
        vbox.addWidget(self.btn3)

        self.btntr = QPushButton('OCR', self)
        self.btntr.setCheckable(True)
        self.btntr.clicked.connect(self.wordRead)
        vbox.addWidget(self.btntr)

        self.btnlib = QPushButton('물체 인식', self)
        self.btnlib.setCheckable(True)
        self.btnlib.clicked.connect(self.objectRead)
        vbox.addWidget(self.btnlib)




        txt = ['얼굴', '눈', '안경']
        self.grp = QButtonGroup(self)
        self.grp = QButtonGroup(self)
        for i in range(len(txt)):
            btn = QCheckBox(txt[i], self)
            self.grp.addButton(btn, i)
            vbox.addWidget(btn)
        vbox.addStretch(1)
        self.grp.setExclusive(False)
        self.grp.buttonClicked[int].connect(self.detectOption)
        self.bDetect = [False for i in range(len(txt))]


        self.btn2 = QPushButton('프로그램 종료', self)
        #self.btn2.clicked.connect(QWidget.close) #처리 속도가 느림
        self.btn2.clicked.connect(CWidget.destroy)
        vbox.addWidget(self.btn2)






        # video area
        self.frm = QLabel(self)
        self.frm.setFrameShape(QFrame.Panel)

        hbox = QHBoxLayout()
        hbox.addLayout(vbox)
        hbox.addWidget(self.frm, 1)
        self.setLayout(hbox)

        self.setFixedSize(size)
        self.move(100,100)
        self.setWindowTitle('기말 레포트')
        self.show()


    def onoffCam(self, e):
        if self.btn.isChecked():
            self.btn.setText('카메라 끄기')
            self.video.startCam()
        else:
            self.btn.setText('카메라 켜기')
            self.video.stopCam()



    def wordRead(self,zk):
        if self.btntr.isChecked():
            self.btntr.setText('캡처 종료')
            self.video.startword()
        else:
            self.btntr.setText('OCR')
            self.video.stopCam()

    def objectRead(self,zk):
        if self.btnlib.isChecked():
            self.btnlib.setText('캠 종료')
            self.video.startlib()
        else:
            self.btnlib.setText('물체 인식')
            self.video.stoplib()

    def locking(self,lr):
        if self.btn3.isChecked():
            self.btn3.setText('캠 종료')
            self.video.startlockj()
        else:
            self.btn3.setText('잠금 해제')
            self.video.jatstop()

    def detectOption(self, id):
        if self.grp.button(id).isChecked():
            self.bDetect[id] = True
        else:
            self.bDetect[id] = False
        #print(self.bDetect)
        self.video.setOption(self.bDetect)

    def recvImage(self, img):
        self.frm.setPixmap(QPixmap.fromImage(img))

    def closeEvent(self, e):
        self.video.stopCam()

 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = CWidget()
    sys.exit(app.exec_())
