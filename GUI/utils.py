from math import sqrt
from GUI.ustr import ustr
import hashlib
import re
import sys
import os
import os.path as osp
from GUI.color import *
import GUI.shape as guishape
import json

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
QT5 = True


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

VISDRONE_CLASSES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
    "others",
)

def new_icon(icon):
    return QIcon(':/' + icon)


def new_button(text, icon=None, slot=None):
    b = QPushButton(text)
    if icon is not None:
        b.setIcon(new_icon(icon))
    if slot is not None:
        b.clicked.connect(slot)
    return b


def new_action(parent, text, slot=None, shortcut=None, icon=None,
               tip=None, checkable=False, enabled=True):
    """Create a new action and assign callbacks, shortcuts, etc."""
    a = QAction(text, parent)
    if icon is not None:
        a.setIcon(new_icon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    return a


def add_actions(widget, actions):
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)


def label_validator():
    return QRegExpValidator(QRegExp(r'^[^ \t].+'), None)


class Struct(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())


def format_shortcut(text):
    mod, key = text.split('+', 1)
    return '<b>%s</b>+<b>%s</b>' % (mod, key)


def generate_color_by_text(text):
    if text == "pedestrian":
        return PERSON_LINE_COLOR, PERSON_FILL_COLOR
    elif text == "people":
        return PEOPLE_LINE_COLOR, PEOPLE_FILL_COLOR
    elif text == "bicycle":
        return BICYCLE_LINE_COLOR, BICYCLE_FILL_COLOR
    elif text == "car":
        return CAR_LINE_COLOR, CAR_FILL_COLOR
    elif text == "van":
        return VAN_LINE_COLOR, VAN_FILL_COLOR
    elif text == "truck":
        return TRUCK_LINE_COLOR, TRUCK_FILL_COLOR
    elif text == "tricycle":
        return TRICYCLE_LINE_COLOR, TRICYCLE_FILL_COLOR
    elif text == "awning-tricycle":
        return AWNING_TRICYCLE_LINE_COLOR, AWNING_TRICYCLE_FILL_COLOR
    elif text == "bus":
        return BUS_LINE_COLOR, BUS_FILL_COLOR
    elif text == "motor":
        return MOTOR_LINE_COLOR, MOTOR_FILL_COLOR  
    elif text == "others" or text == "liangzai":
        return OTHER_LINE_COLOR, OTHER_FILL_COLOR 

    s = ustr(text)
    hash_code = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)
    r = int((hash_code / 255) % 255)
    g = int((hash_code / 65025) % 255)
    b = int((hash_code / 16581375) % 255)
    return QColor(r, g, b, 100)


def have_qstring():
    """p3/qt5 get rid of QString wrapper as py3 has native unicode str type"""
    return not (sys.version_info.major >= 3 or QT_VERSION_STR.startswith('5.'))


def util_qt_strlistclass():
    return QStringList if have_qstring() else list


def natural_sort(list, key=lambda s:s):
    """
    Sort the list into natural alphanumeric order.
    """
    def get_alphanum_key_func(key):
        convert = lambda text: int(text) if text.isdigit() else text
        return lambda s: [convert(c) for c in re.split('([0-9]+)', key(s))]
    sort_key = get_alphanum_key_func(key)
    list.sort(key=sort_key)

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

# QT4 has a trimmed method, in QT5 this is called strip
if QT5:
    def trimmed(text):
        return text.strip()
else:
    def trimmed(text):
        return text.trimmed()


def get_xywhid(shape):

    min_x = sys.maxsize
    min_y = sys.maxsize
    max_x = 0
    max_y = 0

    for point in shape.points:

        min_x = round(min(min_x, point.x()))
        min_y = round(min(min_y, point.y()))
        max_x = round(max(max_x, point.x()))
        max_y = round(max(max_y, point.y()))
    
    w = max_x - min_x
    h = max_y - min_y
    classId = VISDRONE_CLASSES.index(shape.label)

    return (min_x, min_y, w, h, classId)


def generate_coco_json(shapes):

    raise NotImplementedError


def generate_yolo_txts(shapes, width, height, num_frames, Savedir):

    frames = ['' for _ in range(num_frames)]

    for shape in shapes:

        min_x, min_y, w, h, classid = get_xywhid(shape)
        center_x = (min_x + (w / 2)) / width
        center_y = (min_y + (h / 2)) / height
        w = w / width
        h = h / height

        if shape.auto == guishape.STATIONARY_OBJECT:
            for i in range(num_frames):
                frames[i] += f"{classid}, {shape.id}, {center_x}, {center_y}, {w}, {h}\n"
        else:
            frames[shape.frameId-1] += f"{classid}, {shape.id}, {center_x}, {center_y}, {w}, {h}\n"
        
    for i in range(num_frames):
        if frames[i]:
            with open(os.path.join(Savedir, str(i).zfill(8)+'.txt'), 'w') as f:
                f.writelines(frames[i])
                print(f"save {i}th frame label")

def generate_visdrone_txts(shapes, num_frames, savedPath):

    results = []
    for shape in shapes:
        min_x = sys.maxsize
        min_y = sys.maxsize
        max_x = 0
        max_y = 0
        for point in shape.points:
            min_x = round(min(min_x, point.x()))
            min_y = round(min(min_y, point.y()))
            max_x = round(max(max_x, point.x()))
            max_y = round(max(max_y, point.y()))
        w = max_x - min_x
        h = max_y - min_y
        classId = VISDRONE_CLASSES.index(shape.label)
        if shape.auto == guishape.STATIONARY_OBJECT:
            for i in range(1, num_frames + 1):
                results.append(
                f"{i},{shape.id},{min_x},{min_y},{w},{h},{shape.score:.2f},{classId},0,0\n"
            )
        else:
            results.append(
                f"{shape.frameId},{shape.id},{min_x},{min_y},{w},{h},{shape.score:.2f},{classId},0,0\n"
            )
        
    with open(savedPath, 'w') as f:
        f.writelines(results)
        print(f"save results to {savedPath}")