import sys
import string
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QDialog, QMessageBox


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(442, 339)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(60, 280, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayoutWidget = QtWidgets.QWidget(Dialog)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(30, 40, 371, 241))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 2, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 4, 0, 1, 1)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.gridLayout.addWidget(self.lineEdit_4, 3, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 4, 1, 1)
        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout.addWidget(self.lineEdit_3, 1, 2, 1, 1)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.gridLayout.addWidget(self.lineEdit_5, 1, 4, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 1, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 3, 0, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 3, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 0, 5, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 1, 3, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 3, 3, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 0, 1, 1, 1)
        self.lineEdit_6 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.gridLayout.addWidget(self.lineEdit_6, 1, 5, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 5, 0, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 4, 3, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 5, 3, 1, 1)
        self.lineEdit_7 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.gridLayout.addWidget(self.lineEdit_7, 3, 4, 1, 1)
        self.lineEdit_8 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.gridLayout.addWidget(self.lineEdit_8, 3, 5, 1, 1)
        self.lineEdit_9 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.gridLayout.addWidget(self.lineEdit_9, 4, 1, 1, 1)
        self.lineEdit_10 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.gridLayout.addWidget(self.lineEdit_10, 4, 2, 1, 1)
        self.lineEdit_11 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.gridLayout.addWidget(self.lineEdit_11, 5, 1, 1, 1)
        self.lineEdit_12 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_12.setObjectName("lineEdit_12")
        self.gridLayout.addWidget(self.lineEdit_12, 5, 2, 1, 1)
        self.lineEdit_13 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_13.setObjectName("lineEdit_13")
        self.gridLayout.addWidget(self.lineEdit_13, 4, 4, 1, 1)
        self.lineEdit_14 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_14.setObjectName("lineEdit_14")
        self.gridLayout.addWidget(self.lineEdit_14, 4, 5, 1, 1)
        self.lineEdit_15 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_15.setObjectName("lineEdit_15")
        self.gridLayout.addWidget(self.lineEdit_15, 5, 4, 1, 1)
        self.lineEdit_16 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_16.setObjectName("lineEdit_16")
        self.gridLayout.addWidget(self.lineEdit_16, 5, 5, 1, 1)
        self.lineEdit_17 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_17.setObjectName("lineEdit_17")
        self.gridLayout.addWidget(self.lineEdit_17, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_4.setText(_translate("Dialog", "y"))
        self.label_9.setText(_translate("Dialog", "source point3"))
        self.label_3.setText(_translate("Dialog", "x"))
        self.label.setText(_translate("Dialog", "source point1"))
        self.label_2.setText(_translate("Dialog", "source point2"))
        self.label_8.setText(_translate("Dialog", "y"))
        self.label_5.setText(_translate("Dialog", "ground poin1t"))
        self.label_6.setText(_translate("Dialog", "ground point2"))
        self.label_7.setText(_translate("Dialog", "x"))
        self.label_10.setText(_translate("Dialog", "source point4"))
        self.label_11.setText(_translate("Dialog", "ground point3"))
        self.label_12.setText(_translate("Dialog", "ground point4"))


class CoordDialog(QDialog, Ui_Dialog):
    # _signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super(CoordDialog, self).__init__()
        self.setupUi(self)
        self.checked = False
        self.sx1 = None
        self.sx2 = None
        self.sx3 = None
        self.sx4 = None

        self.sy1 = None
        self.sy2 = None
        self.sy3 = None
        self.sy4 = None

        self.dx1 = None
        self.dx2 = None
        self.dx3 = None
        self.dx4 = None

        self.dy1 = None
        self.dy2 = None
        self.dy3 = None
        self.dy4 = None
        self.buttonBox.accepted.connect(self.check_save_input)

    def check_save_input(self):
        try:
            self.sx1 = int(self.lineEdit.text())
            self.sx2 = int(self.lineEdit_2.text())
            self.sx3 = int(self.lineEdit_9.text())
            self.sx4 = int(self.lineEdit_11.text())

            self.sy1 = int(self.lineEdit_3.text())
            self.sy2 = int(self.lineEdit_4.text())
            self.sy3 = int(self.lineEdit_10.text())
            self.sy4 = int(self.lineEdit_12.text())

            self.dx1 = int(self.lineEdit_5.text())
            self.dx2 = int(self.lineEdit_7.text())
            self.dx3 = int(self.lineEdit_13.text())
            self.dx4 = int(self.lineEdit_15.text())

            self.dy1 = int(self.lineEdit_6.text())
            self.dy2 = int(self.lineEdit_8.text())
            self.dy3 = int(self.lineEdit_14.text())
            self.dy4 = int(self.lineEdit_16.text())
            self.checked = True
        except ValueError:
            # TODO find out a better way of doing this.
            flag = QMessageBox.information(self, "Warning", "detected invalid number; ",
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

    def gather_mappings(self):
        """allocate and gather all the user input to the stash.

        Returns:
            src_points (numpy.ndarray): 4 source points.
            dst_points (numpy.ndarray): 4 dest points.
        """
        # TODO figure out a better solution about this.

        src_points = np.float32(
            [[self.sx1, self.sy1], [self.sx2, self.sy2], [self.sx3, self.sy3], [self.sx4, self.sy4]])
        dst_points = np.float32(
            [[self.dx1, self.dy1], [self.dx2, self.dy2], [self.dx3, self.dy3], [self.dx4, self.dy4]])

        # src_points = np.float32([[1, 1], [2, 2], [3, 3], [4, 4]])
        # dst_points = np.float32([[1, 1], [2, 2], [3, 3], [4, 4]])
        print(src_points)

        print(dst_points)
        return src_points, dst_points


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CoordDialog()
    window.show()
    sys.exit(app.exec_())
