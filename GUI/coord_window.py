import sys
import string
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QDialog
from Ui_coord_window import Ui_Dialog


class CoordDialog(QDialog, Ui_Dialog):
    # _signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super(CoordDialog, self).__init__()
        self.setupUi(self)

        # self.retranslateUi(self)
        # self.pushButton.clicked.connect(self.slot1)

    # def format_strings(self):
    #     data_str = self.lineEdit.text()
    #     # 发送信号
    #     self._signal.emit(data_str)

    def gather_mappings(self):
        """allocate and gather all the user input to the stash.

        Returns:
            numpy.ndarray: 4 source points and 4 ground points.
        """
        # TODO figure out a better solution about this.
        try:
            sx1 = string.atoi(self.lineEdit.text())
            sx2 = string.atoi(self.lineEdit2.text())
            sx3 = string.atoi(self.lineEdit9.text())
            sx4 = string.atoi(self.lineEdit11.text())

            sy1 = string.atoi(self.lineEdit3.text())
            sy2 = string.atoi(self.lineEdit4.text())
            sy3 = string.atoi(self.lineEdit10.text())
            sy4 = string.atoi(self.lineEdit12.text())

            dx1 = string.atoi(self.lineEdit5.text())
            dx2 = string.atoi(self.lineEdit7.text())
            dx3 = string.atoi(self.lineEdit13.text())
            dx4 = string.atoi(self.lineEdit15.text())

            dy1 = string.atoi(self.lineEdit6.text())
            dy2 = string.atoi(self.lineEdit8.text())
            dy3 = string.atoi(self.lineEdit14.text())
            dy4 = string.atoi(self.lineEdit16.text())

            src_points = np.float32(
                [[sx1, sy1], [sx2, sy2], [sx3, sy3], [sx4, sy4]])
            dst_points = np.float32(
                [[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
        except TypeError:
            # TODO find out a better way of doing this.
            src_points = np.float32(
                [[1, 1], [2, 2], [3, 3], [4, 4]])
            dst_points = np.float32(
                [[1, 1], [2, 2], [3, 3], [4, 4]])
        return src_points, dst_points


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CoordDialog()
    window.show()
    sys.exit(app.exec_())
