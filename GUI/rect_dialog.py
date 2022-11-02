from PyQt5.QtGui import QCursor, QRegExpValidator
from PyQt5.QtCore import QPoint, Qt, QRegExp
from PyQt5.QtWidgets import QDialog, QLabel, QHBoxLayout, QVBoxLayout, QDialogButtonBox, QLineEdit, QCheckBox

from GUI.utils import new_icon

BB = QDialogButtonBox

class RectifyDialog(QDialog):
    def __init__(self, parent=None):
        super(RectifyDialog, self).__init__(parent)
        self.setWindowTitle("轨迹修正")

        self.toFrame = QLabel("ToFrame: ", self)
        self.frameEdit = QLineEdit(self)
        self.frameEdit.setText("0")
        self.frameEdit.setValidator(QRegExpValidator(QRegExp(r'^[0-9]*[1-9][0-9]*$')))
        hlayoutFrame = QHBoxLayout()
        hlayoutFrame.addWidget(self.toFrame)
        hlayoutFrame.addWidget(self.frameEdit)

        self.targetId = QLabel("TargetId:  ", self)
        self.targetEdit = QLineEdit(self)
        self.targetEdit.setText("0")
        hlayoutTarget = QHBoxLayout()
        hlayoutTarget.addWidget(self.targetId)
        hlayoutTarget.addWidget(self.targetEdit)
        self.frameEdit.setValidator(QRegExpValidator(QRegExp(r'^[0-9]*[1-9][0-9]*$')))

        layout = QVBoxLayout()
        layout.addLayout(hlayoutFrame)
        layout.addLayout(hlayoutTarget)
        self.checkbox = QCheckBox("填充缺失目标的帧", self)
        layout.addWidget(self.checkbox)

        self.button_box = bb = BB(BB.Ok | BB.Cancel, Qt.Horizontal, self)
        bb.button(BB.Ok).setIcon(new_icon('done'))
        bb.button(BB.Cancel).setIcon(new_icon('undo'))
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        self.setLayout(layout)

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
        return (int(self.frameEdit.text()), int(self.targetEdit.text()), self.checkbox.isChecked()) \
            if self.exec_() else (None, None, None)