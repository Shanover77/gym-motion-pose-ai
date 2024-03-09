import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen
import math


class WatchWidget(QWidget):
    def __init__(self, angle, parent=None):
        super(WatchWidget, self).__init__(parent)
        self.angle = angle
        self.setFixedSize(200, 200)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawEllipse(10, 10, 180, 180) # Draw the watch face

        painter.setPen(QPen(Qt.black, 2))
        x2 = round(100 + 100 * self.cos(self.angle))
        y2 = round(100 - 100 * self.sin(self.angle))
        painter.drawLine(100, 100, int(x2), int(y2)) # Draw the needle with rounded and converted coordinates


    def cos(self, angle):
        return self.cos_sin(angle)[0]

    def sin(self, angle):
        return self.cos_sin(angle)[1]

    def cos_sin(self, angle):
        angle = angle * (3.141592653589793 / 180)
        return (math.cos(angle), math.sin(angle))

class MainWindow(QMainWindow):
    def __init__(self, angles, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("Watch Dials")

        self.widget = QWidget()
        self.layout = QGridLayout(self.widget)
        self.setCentralWidget(self.widget)

        for i, angle in enumerate(angles):
            watch = WatchWidget(angle)
            self.layout.addWidget(watch, i // 4, i % 4)

if __name__ == "__main__":
    angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
    app = QApplication(sys.argv)
    mainWin = MainWindow(angles)
    mainWin.show()
    sys.exit(app.exec_())
