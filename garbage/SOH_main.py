import sys
import os

from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.uic.properties import QtGui
import icons
from main import MainWindow
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import ic_model, ic_getin
import re

class second_MainWindow(QMainWindow, MainWindow):
    def __init__(self):
        super(second_MainWindow, self).__init__()
        self.setupUi(self)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = second_MainWindow()
    main.show()
    # app.installEventFilter(main)
    sys.exit(app.exec_())


