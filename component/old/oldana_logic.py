import base64

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QIcon

from src import icons
from component.old.oldana import Ui_Old
import component.old.oldana_import as oldana_import
from PyQt5.QtWidgets import *

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class oldMainWindow(QWidget, Ui_Old):
    def __init__(self):
        super(oldMainWindow, self).__init__()
        self.setupUi(self)

        self.indata = Figure(figsize=(1000, 300), dpi=70)
        self.indata.patch.set_facecolor('none')
        self.indata.patch.set_alpha(0)

        self.lamd = Figure(figsize=(1000, 300), dpi=70)
        self.lamd.patch.set_facecolor('none')

        self.can_indata = FigureCanvas(self.indata)
        self.can_lam = FigureCanvas(self.lamd)

        self.toolbar_in = NavigationToolbar(self.can_indata, self)
        self.toolbar_lam = NavigationToolbar(self.can_lam, self)

        self.verticalLayout.addWidget(self.toolbar_in)
        self.verticalLayout.addWidget(self.can_indata)
        self.verticalLayout_3.addWidget(self.toolbar_lam)
        self.verticalLayout_3.addWidget(self.can_lam)

        self.choose_1.clicked.connect(lambda: self.choosefile(1))
        self.choose_2.clicked.connect(lambda: self.choosedir(2))

        self.analyse.clicked.connect(self.olny_function)

        self.pushButton.clicked.connect(self.deletepic)

    def olny_function(self):
        dataformat = self.format.currentText()
        ic_path = self.path_1.text()
        store_path = self.path_2.text()
        icpeak = self.icpeak.text()
        if(ic_path and store_path and icpeak):
            print("ok")
            try:
                icpeak = float(self.icpeak.text())
                self.indata.clf()
                self.can_indata.draw()

                self.lamd.clf()
                self.can_lam.draw()

                ic_data, cycle_num, plot_cycle, LAM = oldana_import.LAM_analysis_wrap(dataformat, ic_path, store_path, icpeak)

                # INPUT
                INDATA = self.indata.add_subplot(111)
                for cycle in plot_cycle:
                    plot_ic = ic_data.iloc[cycle, :]    # plot_ic / 3600
                    INDATA.plot(range(len(plot_ic)), plot_ic, color=plt.cm.viridis(cycle / cycle_num))
                    # INDATA.colorbar(INDATA.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, cycle_num)),ax=ax, label='Cycles',
                    #                ticks=plot_cycle)
                INDATA.set_title('IC data preview')
                INDATA.set_xlabel('Point index')
                INDATA.set_ylabel('Incremental Capacity (Ah/V)')
                # INDATA.show()

                self.can_indata.draw()

                # LAM
                LAMDRAW = self.lamd.add_subplot(111)
                color = np.array([231, 98, 84]) / 255
                LAMDRAW.plot(range(1, len(LAM) + 1), LAM, color=color)
                LAMDRAW.set_title('LAM quantization result')
                LAMDRAW.set_ylabel('LAM (%)')
                LAMDRAW.set_xlabel('cycle index')
                LAMDRAW.legend()
                # LAMDRAW.show()

                self.can_lam.draw()
            except Exception as e:
                # 捕获异常并显示错误消息
                QMessageBox.critical(self, '出错啦', str(e))

        else:
            self.lackdata()

    def lackdata(self):
        # 将base64编码的图像数据解码为字节数据
        icon_bytes = base64.b64decode(icons.icon1)
        # 将字节数据转换为QPixmap
        pixmap = QPixmap()
        pixmap.loadFromData(icon_bytes)
        # 将QPixmap转换为QIcon
        icon = QIcon(pixmap)

        msg_box = QMessageBox()
        msg_box.setWindowIcon(icon)  # 设置窗口图标
        msg_box.setIconPixmap(pixmap)  # 设置消息框图标
        msg_box.setWindowTitle('出错啦')
        msg_box.setText('请检查输入数据（此对话框会自动关闭）')
        # icon = QIcon()
        # icon.addPixmap(QtGui.QPixmap.fromImage(QtGui.QImage.fromData(base64.b64decode(icons.icon1))))
        # msg_box.setWindowIcon(icon)
        msg_box.setIcon(QMessageBox.Information)
        # 创建一个定时器，3秒后关闭消息框
        timer = QTimer(msg_box)
        timer.setSingleShot(True)
        timer.timeout.connect(msg_box.accept)
        timer.start(1500)  # 3000毫秒后关闭消息框
        msg_box.exec_()

    def deletepic(self):
        self.indata.clf()
        self.can_indata.draw()

        self.lamd.clf()
        self.can_lam.draw()


    def choosefile(self, index):
        fname, _ = QFileDialog.getOpenFileName(None, '选择路径')
        if fname:  # 如果用户选择了文件
            fname = str(fname)
            getattr(self, f"path_{index}").setText(fname)


    def choosedir(self, index):
        fname = QFileDialog.getExistingDirectory(None, '选择路径')
        if fname:  # 如果用户选择了文件
            fname = str(fname)
            getattr(self, f"path_{index}").setText(fname)
