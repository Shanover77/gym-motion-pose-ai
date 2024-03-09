from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import random
import sys


class CircularDiagram(QMainWindow):
    def __init__(self):
        super(CircularDiagram, self).__init__()

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout(self.central_widget)

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.main_layout.addWidget(self.canvas)

        self.timer = self.setup_timer()
        self.start_animation()

    def setup_timer(self):
        timer = QTimer(self)
        timer.timeout.connect(self.update_data)
        timer.start(1000)  # Update every 1000 milliseconds (1 second)
        return timer

    def update_data(self):
        new_value = random.uniform(0, 100)
        self.canvas.update_data(new_value)

    def start_animation(self):
        self.canvas.start_animation()

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig, self.ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(fig)
        self.setParent(parent)

        self.theta = np.linspace(0, 2 * np.pi, 1000)
        self.values = []

        self.line, = self.ax.plot([], [], lw=2)

    def start_animation(self):
        self.animation = FuncAnimation(self.figure, self.update_data, interval=100)
        plt.show()

    def update_data(self, new_value):
        self.values.append(new_value)
        if len(self.values) > 1000:
            self.values.pop(0)

def update_plot(self):
        if len(self.theta) == len(self.values):
            self.line.set_data(self.theta, self.values)
        self.line.set_color('b')
        self.line.set_linewidth(2)
        self.line.set_marker('o')
        self.line.set_markersize(8)
        self.line.set_markeredgecolor('r')
        self.line.set_markerfacecolor('none')
        self.ax.set_rmax(2 * np.pi)
        self.ax.set_yticklabels([])
        self.ax.set_xticklabels([])
        self.ax.grid(False)
        self.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = CircularDiagram()
    mainWin.show()
    sys.exit(app.exec_())
