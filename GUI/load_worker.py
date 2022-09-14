import os

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import time

class loadWorker(QThread):
    sinOut = pyqtSignal(str)

    def __init__(self, canvas):
        super().__init__()
        self.labelPath = ""
        self.canvas = canvas

    def run(self):
        self.load_label()

    def set_label_type(self, labelTpye):
        self.labelType = labelTpye

    def load_path(self, labelPath):
        self.labelPath = labelPath

    def load_yolo_cfig(self, labelDir, videoWidth, videoHeight):
        self.labelDir = labelDir
        self.videoWidth = videoWidth
        self.videoHeight = videoHeight

    def load_label(self):
        # Yolo格式 - yolo格式是把每一帧用一个txt文件存储，因此要用一个文件夹来存储和读取
        # Yolo格式只有目标框的位置和帧号信息，如果要补充应该需要手动输入target_id
        if self.labelType == 'Yolo':
            # 需要确保帧号是txt文件的后缀
            import re
            count = len(os.listdir(self.labelDir))
            for i, txt in enumerate(os.listdir(self.labelDir)):
                frameid = int(re.findall("\d+", txt)[-1])
                with open(self.labelDir + os.sep + txt, "r") as f:
                    for line in f.readlines():
                        line = [float(x) for x in line.strip('\n').split(' ')]
                        label = int(line[0])
                        bbox = line[1:]
                        left = int((bbox[0] - bbox[2] / 2.0) * self.videoWidth)
                        top = int((bbox[1] - bbox[3] / 2.0) * self.videoHeight)
                        width = int(bbox[2] * self.videoWidth)
                        height = int(bbox[3] * self.videoHeight)
                        tlwh = [left, top, width, height]
                        self.canvas.update_shape(id=-1, frameId=frameid, cls_id=label, tlwh=tlwh, score=1.0, auto='L')
                if i % 10 == 0:
                    self.sinOut.emit("标注框已加载 {} / {}".format(i, count))
        elif self.labelType == 'Coco':
            pass
        else:
        # VisDrone格式
            with open(self.labelPath, "r") as f:
                count = len(open(self.labelPath,'rU').readlines())
                for i, line in enumerate(f.readlines(), 1):
                    line = line.strip('\n').split(',')
                    tlwh = [int(line[2]), int(line[3]), int(line[4]), int(line[5])]
                    self.canvas.update_shape(int(line[1]), int(line[0]), int(line[7]), tlwh, float(line[6]), 'L')
                    if i % 10 == 0:
                        self.sinOut.emit("标注框已加载 {} / {}".format(i, count))
        self.sinOut.emit("标注文件已加载完成")
