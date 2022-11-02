from PyQt5.QtGui import QCursor
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtWidgets import QDialog, QLabel, QHBoxLayout, QVBoxLayout, QDialogButtonBox, QLineEdit, QComboBox

from GUI.utils import new_icon

BB = QDialogButtonBox

class ConfigDialog(QDialog):
    def __init__(self, parent=None, yaml_path=None):
        super(ConfigDialog, self).__init__(parent)

        self.yaml_path = yaml_path

        self.attrName = QLabel("属性", self)
        self.valName = QLabel("值", self)
        hlayoutHead = QHBoxLayout()
        hlayoutHead.addWidget(self.attrName)
        hlayoutHead.addWidget(self.valName)

        hlayoutItem = QHBoxLayout()
        self.cb = QComboBox()
        self.textEdit = QLineEdit(self)
        hlayoutItem.addWidget(self.cb)
        hlayoutItem.addWidget(self.textEdit)
        self.textEdit.setMinimumWidth(350)

        self.attrList = []
        self.valList = []
        with open(self.yaml_path, 'r') as f:
            for line in f.readlines():
                ls = line.strip('\n').split(':')
                if len(ls) != 2:
                    continue
                name, val = ls
                self.attrList.append(name)
                self.valList.append(val)
                # print(name, val)

        self.cb.addItems(self.attrList)
        self.cb.currentIndexChanged.connect(self.config_combo_selection_changed)
        self.textEdit.setText(self.valList[0])
        self.textEdit.editingFinished.connect(self.config_attr_text_changed)

        layout = QVBoxLayout()
        layout.addLayout(hlayoutHead)
        layout.addLayout(hlayoutItem)
        self.button_box = bb = BB(BB.Ok | BB.Cancel, Qt.Horizontal, self)
        bb.button(BB.Ok).setIcon(new_icon('done'))
        bb.button(BB.Cancel).setIcon(new_icon('undo'))
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        self.setLayout(layout)

    def push_up(self):
        assert len(self.attrList) == len(self.valList)
        with open(self.yaml_path, 'w') as f:
            for i in range(len(self.attrList)):
                f.write(f'{self.attrList[i]}:{self.valList[i]}\n')

    def config_attr_text_changed(self):
        if not self.textEdit.text().startswith(' '):
            self.textEdit.setText(' ' + self.textEdit.text())
        self.valList[self.cb.currentIndex()] = self.textEdit.text()

    def config_combo_selection_changed(self, index):
        self.textEdit.setText(self.valList[index])

    def validate(self):
        self.accept()

    def pop_up(self, move=True):
        if move:
            cursor_pos = QCursor.pos()
            parent_bottom_right = self.parentWidget().geometry()
            max_x = parent_bottom_right.x() + parent_bottom_right.width() - self.sizeHint().width()
            max_y = parent_bottom_right.y() + parent_bottom_right.height() - self.sizeHint().height()
            max_global = self.parentWidget().mapToGlobal(QPoint(max_x, max_y))
            if cursor_pos.x() > max_global.x():
                cursor_pos.setX(max_global.x())
            if cursor_pos.y() > max_global.y():
                cursor_pos.setY(max_global.y())
            self.move(cursor_pos)
        return True if self.exec_() else False