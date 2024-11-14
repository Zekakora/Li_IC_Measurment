import sys

from PyQt5.QtWidgets import QMainWindow, QApplication
from main import MainWindow
import matplotlib

matplotlib.use("Qt5Agg")  # 声明使用QT5


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


