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
        self.pushButton.clicked.connect(self.ic_getin)

        # self.gridlayout = QGridLayout(self.groupBox_2)
        # self.gridlayout.addWidget(self.F, 0, 1)
        # self.gridlayout = QGridLayout(self.groupBox)
        # self.gridlayout.addWidget(self.F, 0, 1)

    # def icgetin(self):
    #     print(self.data_format)
    #     print(self.v_path)
    #     print(self.ic_path)
    #     if self.data_format and self.v_path and self.ic_path:
    #         ic_getin.ic_getin(self.data_format, self.v_path, self.ic_path)
    #     else:
    #         print("Missing information")

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

    def ic_getin(self, data_formate, v_path, ic_path):
         print(self.data_format)
         print(self.v_path)
         print(self.ic_path)
        # # 包含两个数据的预览图
        # if data_formate == 'txt':
        #     ic_data = np.genfromtxt(ic_path, delimiter=',')
        #     ic_data = np.delete(ic_data, -1, axis=1)  # ic数值为单位C/V或者mAh/V
        #     output_size = ic_data.shape[1]
        #     v_data = []
        #     with open(v_path, 'r') as file:
        #         v_data = file.readlines()
        #     cycle_num = len(v_data)
        #     plot_cycle = range(0, cycle_num, int(cycle_num / 10))
        #     fig, ax = plt.subplots()
        #     ##input_figure,Q_V的图像
        #     # plt.figure()
        #     for cycle in plot_cycle:
        #         plot_y = v_data[cycle].split(',')
        #         plot_y = np.array(plot_y)  # 假设以逗号为分隔符
        #         plot_y = np.delete(plot_y, -1, axis=0)
        #         plot_y = plot_y.astype(float)
        #         plt.plot(range(len(plot_y)), plot_y, color=plt.cm.viridis(cycle / cycle_num))
        #     plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, cycle_num)), ax=ax, label='Cycles',
        #                  ticks=plot_cycle)
        #     plt.title('Q-V data preview')
        #     plt.xlabel('Point index')
        #     plt.ylabel('Voltage (V)')
        #     plt.show()
        #
        #     ##Output_figure,IC的图像
        #     fig, ax = plt.subplots()
        #     for cycle in plot_cycle:
        #         plt.plot(range(len(ic_data[cycle])), ic_data[cycle] / 3600, color=plt.cm.viridis(cycle / cycle_num))
        #     plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, cycle_num)), ax=ax, label='Cycles',
        #                  ticks=plot_cycle)
        #     plt.title('IC data preview')
        #     plt.xlabel('Point index')
        #     plt.ylabel('Incremental Capacity (Ah/V)')
        #     plt.show()
        #     # 判断老化机制,使用一个输出框判断
        #     peak_data = ic_data[0]
        #     peaks, _ = find_peaks(abs(peak_data), height=-6 * 3600, threshold=0.1 * 3600)
        #     peaks_num = len(peaks)
        #     if peaks_num == 1:
        #         peak_data = ic_data[-1]
        #         peaks1, _ = find_peaks(abs(peak_data), height=-6 * 3600, threshold=0.1 * 3600)
        #         if peaks1[0] < peaks[0]:
        #             degration = '活性物质损失'
        #         else:
        #             degration = '锂库存和活性物质损失'
        #     elif peaks_num == 2:
        #         degration = '锂库存和活性物质损失'
        #
        # elif data_formate == 'excel':
        #     ic_data = pd.read_excel(ic_path)
        #     output_size = ic_data.shape[1]
        #     v_data = pd.read_excel(v_path)
        #     cycle_num = len(v_data)
        #     plot_cycle = range(0, cycle_num, int(cycle_num / 10))
        #     ##input_figure,Q_V的图像
        #     fig, ax = plt.subplots()
        #     for cycle in plot_cycle:
        #         plot_v = v_data.iloc[cycle, :]
        #         plot_v = [value for value in plot_v if pd.notna(value)]
        #         plt.plot(range(len(plot_v)), plot_v, color=plt.cm.viridis(cycle / cycle_num))
        #     plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, cycle_num)), ax=ax, label='Cycles',
        #                  ticks=plot_cycle)
        #     plt.title('Q-V data preview')
        #     plt.xlabel('Point index')
        #     plt.ylabel('Voltage (V)')
        #     plt.show()
        #
        #     ##Output_figure,IC的图像
        #     fig, ax = plt.subplots()
        #     for cycle in plot_cycle:
        #         plot_ic = ic_data.iloc[cycle, :]
        #         plt.plot(range(len(plot_ic)), plot_ic / 3600, color=plt.cm.viridis(cycle / cycle_num))
        #     plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, cycle_num)), ax=ax, label='Cycles',
        #                  ticks=plot_cycle)
        #     plt.title('IC data preview')
        #     plt.xlabel('Point index')
        #     plt.ylabel('Incremental Capacity (Ah/V)')
        #     plt.show()
        #     # 判断老化机制,使用一个输出框判断
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
        # return v_data, ic_data, output_size, degration

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    #app.installEventFilter(main)
    sys.exit(app.exec_())

