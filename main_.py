from PyQt5.QtWidgets import *
import sys
from licon import Ui_MainWindow
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import definition
import ic_getin
import ic_getii


class MyFigure(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MyFigure, self).__init__(self.fig)  # 此句必不可少，否则不能显示图形
        self.axes = self.fig.add_subplot(111)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.F = MyFigure(width=8, height=4, dpi=100)

        self.data_format = self.comboBox.currentText()
        self.v_path = None
        self.ic_path = None

        self.filebutton_2.clicked.connect(self.choosefile_2)
        self.filebutton_1.clicked.connect(self.choosefile_1)
        self.pushButton.clicked.connect(self.ic_getin_ref)


        self.gridlayout_inup = QGridLayout(self.groupBox_2)
        self.gridlayout_indown = QGridLayout(self.groupBox)

        self.icinup = plt.figure()
        self.icindown = plt.figure()

        self.canvas = FigureCanvas(self.icinup)
        self.canvas_1 = FigureCanvas(self.icindown)

        # 将 FigureCanvas 添加到第一个 Group Box 的布局中
        self.gridlayout_inup.addWidget(self.canvas)
        # 将 FigureCanvas 添加到第二个 Group Box 的布局中
        self.gridlayout_indown.addWidget(self.canvas_1)

        self.pushButton.clicked.connect(self.ic_getin_ref)


    def choosefile_1(self):
        fname, _ = QFileDialog.getOpenFileName(None, '选择文件', '/home')
        if fname:  # 如果用户选择了文件
            fname = str(fname)
            self.filepath_1.setPlainText(fname)
            self.v_path = fname

    def choosefile_2(self):
        fname, _ = QFileDialog.getOpenFileName(None, '选择文件', '/home')
        if fname:  # 如果用户选择了文件
            fname = str(fname)
            self.filepath_2.setPlainText(fname)
            self.ic_path = fname

    def ic_getin(self):
        data_format = self.data_format
        ic_path = str(self.ic_path)
        v_path = str(self.v_path)

        Fup = MyFigure(width=6, height=4, dpi=100)
        Fdown = MyFigure(width=6, height=4, dpi=100)
        # 包含两个数据的预览图
        if data_format == 'Txt':
            ic_data = np.genfromtxt(ic_path, delimiter=',')
            ic_data = np.delete(ic_data, -1, axis=1)  # ic数值为单位C/V或者mAh/V
            output_size = ic_data.shape[1]
            v_data = []
            with open(v_path, 'r') as file:
                v_data = file.readlines()
            cycle_num = len(v_data)
            plot_cycle = range(0, cycle_num, int(cycle_num / 10))

            ##input_figure,Q_V的图像
            # plt.figure()
            Fup = self.icinup.add_subplot(111)
            for cycle in plot_cycle:
                plot_y = v_data[cycle].split(',')
                plot_y = np.array(plot_y)  # 假设以逗号为分隔符
                plot_y = np.delete(plot_y, -1, axis=0)
                plot_y = plot_y.astype(float)
                Fup.plot(range(len(plot_y)), plot_y, color=plt.cm.viridis(cycle / cycle_num))
            # Fup.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, cycle_num)), ax=ax, label='Cycles',ticks=plot_cycle)
            Fup.set_title('Q-V data preview')
            Fup.set_xlabel('Point index')
            Fup.set_ylabel('Voltage (V)')
            Fup.show()

            self.canvas.draw()

            ##Output_figure,IC的图像
            Fdown = self.icindown.add_subplot(111)
            fig, ax = plt.subplots()
            for cycle in plot_cycle:
                plt.plot(range(len(ic_data[cycle])), ic_data[cycle] / 3600, color=plt.cm.viridis(cycle / cycle_num))
            # Fdown.plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, cycle_num)), ax=ax, label='Cycles',ticks=plot_cycle)
            Fdown.set_title('IC data preview')
            Fdown.set_xlabel('Point index')
            Fdown.set_ylabel('Incremental Capacity (Ah/V)')

            self.icindown.tight_layout()
            self.canvas_1.draw()
            # 判断老化机制,使用一个输出框判断
            peak_data = ic_data[0]
            peaks, _ = find_peaks(abs(peak_data), height=-6 * 3600, threshold=0.1 * 3600)
            peaks_num = len(peaks)
            if peaks_num == 1:
                peak_data = ic_data[-1]
                peaks1, _ = find_peaks(abs(peak_data), height=-6 * 3600, threshold=0.1 * 3600)
                if peaks1[0] < peaks[0]:
                    degration = '活性物质损失'
                else:
                    degration = '锂库存和活性物质损失'
            elif peaks_num == 2:
                degration = '锂库存和活性物质损失'



        elif data_format == 'Excel':
            ic_data = pd.read_excel(ic_path)
            output_size = ic_data.shape[1]
            v_data = pd.read_excel(v_path)
            cycle_num = len(v_data)
            plot_cycle = range(0, cycle_num, int(cycle_num / 10))


            ##input_figure,Q_V的图像
            ax = self.icinup.add_subplot(111)
            for cycle in plot_cycle:
                plot_v = v_data.iloc[cycle, :]
                plot_v = [value for value in plot_v if pd.notna(value)]
                ax.plot(range(len(plot_v)), plot_v, color=plt.cm.viridis(cycle / cycle_num))
            # ax.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, cycle_num)), ax=ax, label='Cycles',ticks=plot_cycle)
            ax.set_title('Q-V data preview')
            ax.set_xlabel('Point index')
            ax.set_ylabel('Voltage (V)')

            self.icinup.tight_layout()  # 调整布局以确保子图不会重叠
            self.canvas.draw()  # 重绘图形

            ##Output_figure,IC的图像
            Fdown = self.icindown.add_subplot(111)
            for cycle in plot_cycle:
                plot_ic = ic_data.iloc[cycle, :]
                Fdown.plot(range(len(plot_ic)), plot_ic / 3600, color=plt.cm.viridis(cycle / cycle_num))
            # Fdown.plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, cycle_num)), ax=ax, label='Cycles',ticks=plot_cycle)
            Fdown.set_title('IC data preview')
            Fdown.set_xlabel('Point index')
            Fdown.set_ylabel('Incremental Capacity (Ah/V)')
            self.icindown.tight_layout()
            self.canvas_1.draw()

            # 判断老化机制,使用一个输出框判断
        #     peak_data = ic_data.iloc[0, :]
        #     peaks, _ = find_peaks(abs(peak_data), height=-6 * 3600, threshold=0.1 * 3600)
        #     peaks_num = len(peaks)
        #     if peaks_num == 1:
        #         peak_data = ic_data.iloc[-1, :]
        #         peaks1, _ = find_peaks(abs(peak_data), height=-6 * 3600, threshold=0.1 * 3600)
        #         if peaks1[0] < peaks[0]:
        #             degration = '活性物质损失'
        #         else:
        #             degration = '锂库存和活性物质损失'
        #     elif peaks_num == 2:
        #         degration = '锂库存和活性物质损失'
        return
        # return v_data, ic_data, output_size, degration

    def ic_getin_ref(self):
        if self.v_path and self.ic_path and self.data_format:
            self.ic_getin()
            # v_data, ic_data, output_size, degration = self.ic_getin()
            # print(v_data, ic_data, output_size, degration)
        else:
            print("Wrong file found")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    # app.installEventFilter(main)
    sys.exit(app.exec_())
