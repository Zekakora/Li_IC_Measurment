import sys
from PyQt5.QtWidgets import QApplication
from mainwindow import MainWindow
from plotter import plot_sine_wave
import numpy as np

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()

    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plot_sine_wave(window.figure, x, y)

    window.show()
    sys.exit(app.exec_())
