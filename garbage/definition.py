import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


def set_text(self, fname):
    self.textBrowser.clear()
    fname = str(fname)
    self.textBrowser.append(fname)
    # self.textBrowser.append(fname)


def choosefile(self):
    fname, _ = QFileDialog.getOpenFileName(None, '选择文件', '/home')
    if fname:  # 如果用户选择了文件
        self.textBrowser.clear()
        fname = str(fname)
        self.textBrowser.append(fname)


def evaluation(self):
    max = 5
    min = 2
    ave = 3.5
    self.textBrowser_1.clear()
    result = eval_conbime(max, ave, min)
    self.textBrowser_1.append(result)


def eval_conbime(max, ave, min):
    return f"最大：{max}\t\t平均：{ave}\t\t最小：{min}"

def selectpath(self):
    # 清屏
    plt.cla()
    # 获取绘图并绘制
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlim([-1, 6])
    ax.set_ylim([-1, 6])
    ax.plot([0, 1, 2, 3, 4, 5], 'o--')
    cavans = FigureCanvas(fig)
    # 将绘制好的图像设置为中心 Widget
    self.setCentralWidget(cavans)

def plotcos(self):
    t = np.arange(0.0, 5.0, 0.01)
    s = np.cos(2 * np.pi * t)
    self.F.axes.plot(t, s)
    self.F.fig.suptitle("cos")


from garbage.licon import Ui_MainWindow

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


