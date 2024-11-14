#-*-coding:utf-8-*-
from PyQt5.QtWidgets import *
import sys
import numpy as np
from garbage.licon import Ui_MainWindow
import definition
import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


#创建一个matplotlib图形绘制类
class MyFigure(FigureCanvas):
    def __init__(self,width=5, height=4, dpi=100):
        #第一步：创建一个创建Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        #第二步：在父类中激活Figure窗口
        super(MyFigure,self).__init__(self.fig) #此句必不可少，否则不能显示图形
        #第三步：创建一个子图，用于绘制图形用，111表示子图编号，如matlab的subplot(1,1,1)
        self.axes = self.fig.add_subplot(111)
    #第四步：就是画图，【可以在此类中画，也可以在其它类中画】
    def plotsin(self):
        self.axes0 = self.fig.add_subplot(111)
        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2 * np.pi * t)
        self.axes0.plot(t, s)


class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        self.setupUi(self)


        #第五步：定义MyFigure类的一个实例
        self.F = MyFigure(width=8, height=4, dpi=100)
        self.F.plotsin()
        definition.plotcos(self)
        #第六步：在GUI的groupBox中创建一个布局，用于添加MyFigure类的实例（即图形）后其他部件。
        self.gridlayout = QGridLayout(self.groupBox_2)  # 继承容器groupBox
        self.gridlayout.addWidget(self.F,0,1)

        self.gridlayout_1 = QGridLayout(self.groupBox)
        self.plotother()

        self.pushButton.clicked.connect(self.choosefile) # type: ignore
        # self.filebutton_1.clicked.connect(self.selectpath) # type: ignore
        self.filebutton_2.clicked.connect(self.selectpath) # type: ignore

    def selectpath(self):

        # 打开文件夹选择对话框
        folder_dialog = QFileDialog.getExistingDirectory(self, f"选择文件夹", "", QFileDialog.ShowDirsOnly)

        # 如果用户选择了文件夹，则更新 QLabel 中的文本
        if folder_dialog:
            self.filepath_2.setPlainText(folder_dialog)

    def choosefile(self):
        fname, _ = QFileDialog.getOpenFileName(None, '选择文件', '/home')
        if fname:  # 如果用户选择了文件
            fname = str(fname)
            self.filepath_1.setPlainText(fname)

    def plotcos(self):
        t = np.arange(0.0, 5.0, 0.01)
        s = np.cos(2 * np.pi * t)
        self.F.axes.plot(t, s)
        self.F.fig.suptitle("cos")

    def plotother(self):
        F1 = MyFigure(width=6, height=4, dpi=100)
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        y = [23, 21, 32, 13, 3, 132, 13, 3, 1]
        F1.axes.plot(x, y)
        F1.axes.set_title("line")
        self.gridlayout_1.addWidget(F1, 0, 1)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    #app.installEventFilter(main)
    sys.exit(app.exec_())