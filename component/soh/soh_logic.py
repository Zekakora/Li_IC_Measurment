import base64

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import *
import os

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from src import icons
from component.soh.licon_soh import Ui_Form
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# import icsoh, ic_getin
# import soh_getin, soh_model
# import soh_getin, soh_model
import component.soh.soh_model as soh_model
import component.soh.soh_getin as soh_getin
import re


# 定义一个函数，用于从文件名中提取数字部分
def extract_number(filename):
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        return -1  # 如果找不到数字，则返回一个标记值

class MyFigure(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MyFigure, self).__init__(self.fig)  # 此句必不可少，否则不能显示图形
        self.axes = self.fig.add_subplot(111)


# IC
class sohMainWindow(QWidget, Ui_Form):
    def __init__(self):

        super(sohMainWindow, self).__init__()
        self.setupUi(self)
        self.F = MyFigure(width=8, height=4, dpi=100)

        # 选项框内容定义区
        # self.data_format = self.comboBox.currentText()
        self.v_path = None
        self.v_data = None
        self.ic_data = None
        self.ic_path = None
        self.degration = None
        self.output_size = None

        self.labels = None

        self.window_size = None
        self.epoch_num = None
        self.batch_size = None
        self.train_ratio = None
        self.model_select = self.comboBox.currentText()
        self.best_model_path = None
        self.best_model_name = None
        self.parameter_path = None

        self.total_loss = None
        self.total_vaildloss = None

        # 文件获取按钮
        self.filebutton_1.clicked.connect(lambda: self.choosefile_old(1))
        self.filebutton_2.clicked.connect(lambda: self.choosefile_old(2))
        self.filebutton_3.clicked.connect(lambda: self.choosefile(3))
        self.filebutton_4.clicked.connect(lambda: self.choosefile(4))
        # self.filebutton_5.clicked.connect(lambda: self.choosefile(5))
        self.filebutton_6.clicked.connect(lambda: self.choosefile_old(6))
        self.filebutton_7.clicked.connect(lambda: self.choosefile_old(7))
        self.filebutton_8.clicked.connect(lambda: self.choosefile(8))

        self.pushButton_2.clicked.connect(self.clean)

        # 输入数据展示按钮
        self.pushButton.clicked.connect(self.ic_getin_ref)

        # 输入数据绘图
        self.gridlayout_inup = QGridLayout(self.groupBox_2)
        self.gridlayout_indown = QGridLayout(self.groupBox)

        self.icinup = plt.figure()
        self.icindown = plt.figure()

        self.canvas = FigureCanvas(self.icinup)
        self.canvas_1 = FigureCanvas(self.icindown)

        self.toolbar_canva = NavigationToolbar(self.canvas, self)
        self.toolbar_canva_1 = NavigationToolbar(self.canvas_1, self)

        # 将 FigureCanvas 添加到第一个 Group Box 的布局中
        self.gridlayout_inup.addWidget(self.toolbar_canva)
        self.gridlayout_inup.addWidget(self.canvas)
        # 将 FigureCanvas 添加到第二个 Group Box 的布局中
        self.gridlayout_indown.addWidget(self.toolbar_canva_1)
        self.gridlayout_indown.addWidget(self.canvas_1)
        # self.pushButton.clicked.connect(self.ic_getin_ref)

        # loss图绘制
        self.icloss = Figure(figsize=(5, 4), dpi=100)
        self.canvas_loss = FigureCanvas(self.icloss)
        self.toolbar_loss = NavigationToolbar(self.canvas_loss, self)
        self.verticalLayout.addWidget(self.toolbar_loss)
        self.verticalLayout.addWidget(self.canvas_loss)

        # # MAE
        # self.mae_rmse = Figure(figsize=(2, 1.5), dpi=70)
        # self.mae_rmse1 = Figure(figsize=(2, 1.5), dpi=70)
        # self.mae_rmse2 = Figure(figsize=(2, 1.5), dpi=70)
        # self.mae_rmse3 = Figure(figsize=(2, 1.5), dpi=70)
        # self.canva_mae_best = FigureCanvas(self.mae_rmse)
        # self.canva_mae_worst = FigureCanvas(self.mae_rmse1)
        # self.horizontalLayout_3.addWidget(self.canva_mae_best)
        # self.horizontalLayout_3.addWidget(self.canva_mae_worst)
        # self.canva_rmse_best = FigureCanvas(self.mae_rmse2)
        # self.canva_rmse_worst = FigureCanvas(self.mae_rmse3)
        # self.horizontalLayout_5.addWidget(self.canva_rmse_best)
        # self.horizontalLayout_5.addWidget(self.canva_rmse_worst)

        self.groundvspred = Figure(figsize=(5, 4), dpi=80)
        self.truevspred = Figure(figsize=(5, 4), dpi=80)
        self.canva_gvp = FigureCanvas(self.groundvspred)
        self.canva_tvp = FigureCanvas(self.truevspred)

        self.toolbar_gvp = NavigationToolbar(self.canva_gvp, self)
        self.toolbar_tvp = NavigationToolbar(self.canva_tvp, self)
        self.verticalLayout_6.addWidget(self.toolbar_gvp)
        self.verticalLayout_6.addWidget(self.canva_gvp)

        self.verticalLayout_7.addWidget(self.toolbar_tvp)
        self.verticalLayout_7.addWidget(self.canva_tvp)

        # 模型训练窗口
        self.pushButton_3.clicked.connect(self.train_model)
        # self.status_label.setText("已成功启动窗口")
        #   模型测试
        self.pushButton_4.clicked.connect(self.test_model)
        # 模型预测
        self.pushButton_5.clicked.connect(self.predict_model)
        self.icloss.patch.set_facecolor('none')
        self.icindown.patch.set_facecolor('none')
        self.icinup.patch.set_facecolor('none')
        self.groundvspred.patch.set_facecolor('none')
        self.truevspred.patch.set_facecolor('none')

    def choosefile_old(self, index):
        fname, _ = QFileDialog.getOpenFileName(None, '选择文件')
        if fname:  # 如果用户选择了文件
            fname = str(fname)
            getattr(self, f"filepath_{index}").setPlainText(fname)
            setattr(self, f"ic_path_{index}", fname)
            # self.status_label.setText("已选择文件{}".format(index))

    def choosefile(self, index):
        fname = QFileDialog.getExistingDirectory(None, '选择路径')
        if fname:  # 如果用户选择了文件
            fname = str(fname)
            getattr(self, f"filepath_{index}").setPlainText(fname)
            setattr(self, f"ic_path_{index}", fname)
            # self.status_label.setText("已选择路径{}".format(index))

    def ic_getin(self):
        data_format = self.data_format
        ic_path = str(self.ic_path)
        v_path = str(self.v_path)
        eol_num = float(self.eol.text())
        nominal_capacity = float(self.nomi_capa.text())

        eol = eol_num * nominal_capacity / 100
        print("开始加载数据 当前数据格式:{}".format(self.data_format))
        # self.status_label.setText("开始加载数据 当前数据格式:{}".format(self.data_format))
        output_size = None
        self.icinup.clf()
        self.icindown.clf()
        Fup = MyFigure(width=6, height=4, dpi=100)
        Fdown = MyFigure(width=6, height=4, dpi=100)

        # 包含两个数据的预览图
        if data_format == 'TXT':
            ic_data = np.genfromtxt(ic_path, delimiter=',')
            ic_data = np.delete(ic_data, -1, axis=1)  # ic数值为单位C/V或者mAh/V
            soh_data = (ic_data - eol) / (nominal_capacity - eol) * 100
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
            self.canvas.draw()

            ##Output_figure,IC的图像
            Fdown = self.icindown.add_subplot(111)
            fig, ax = plt.subplots()
            norm = Normalize(vmin=soh_data.min(), vmax=soh_data.max())
            cmap = plt.get_cmap('seismic')

            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # 只是为了满足colorbar的需要
            for i in range(cycle_num - 1):
                Fdown.plot(range(1,cycle_num+1)[i:i + 2], soh_data[i:i + 2], color=sm.to_rgba(soh_data[i]))
            # Fdown.plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, cycle_num)), ax=ax, label='Cycles',ticks=plot_cycle)
            Fdown.set_title('SOH data preview')
            Fdown.set_xlabel('cycle index')
            Fdown.set_ylabel('SOH (%)')
            self.icindown.tight_layout()
            self.canvas_1.draw()


        elif data_format == 'Excel':
            capacity_data = pd.read_excel(ic_path)
            soh_data = (capacity_data - eol) / (nominal_capacity - eol) * 100
            output_size = soh_data.shape[1]
            v_data = pd.read_excel(v_path)
            cycle_num = len(v_data)
            plot_cycle = range(0, cycle_num, int(cycle_num / 10))

            ##input_figure,Q_V的图像
            # plt.figure()
            Fup = self.icinup.add_subplot(111)
            for cycle in plot_cycle:
                plot_v = v_data.iloc[cycle, :]
                plot_v = [value for value in plot_v if pd.notna(value)]
                Fup.plot(range(len(plot_v)), plot_v, color=plt.cm.seismic(cycle / cycle_num))

            # Fup.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, cycle_num)), ax=ax, label='Cycles',ticks=plot_cycle)
            Fup.set_title('Q-V data preview')
            Fup.set_xlabel('Point index')
            Fup.set_ylabel('Voltage (V)')

            self.canvas.draw()

            ##Output_figure,IC的图像
            Fdown = self.icindown.add_subplot(111)
            fig, ax = plt.subplots()
            norm = Normalize(vmin=soh_data.min(), vmax=soh_data.max())
            cmap = plt.get_cmap('seismic')
            # 创建ScalarMappable对象
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # 只是为了满足colorbar的需要
            # 绘制线，颜色随soh值变化
            for i in range(cycle_num - 1):
                Fdown.plot(range(1, cycle_num + 1)[i:i + 2], soh_data[i:i + 2])  # , color=sm.to_rgba(soh_data[i]))
            # 添加colorbar
            # cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
            # cbar.set_label('SOH value')
            Fdown.set_title('SOH data preview')
            Fdown.set_xlabel('cycle index')
            Fdown.set_ylabel('SOH (%)')

            self.icindown.tight_layout()
            self.canvas_1.draw()

        self.v_data = v_data
        self.ic_data = soh_data
        self.output_size = output_size

        # print(self.output_size)

        return
        # return v_data, ic_data, output_size, degration

    # 检查并获取ic文件
    def ic_getin_ref(self):
        self.v_path = self.filepath_1.toPlainText()
        self.ic_path = self.filepath_2.toPlainText()
        self.data_format = self.comboBox.currentText()
        # self.status_label.setText("开始加载数据 当前数据格式:{}".format(self.data_format))
        if self.v_path and self.ic_path and self.data_format:
            try:
                self.ic_getin()
                # self.status_label.setText("已成功加载数据".format(self.data_format))
                # v_data, ic_data, output_size, degration = self.ic_getin()
                # print(v_data, ic_data, output_size, degration)
            except Exception as e:
                # 捕获异常并显示错误消息
                QMessageBox.critical(self, '出错啦', str(e))

        else:
            self.lackdata()
            # self.status_label.setText("请检查输入的数据")

    def train_model(self):
        self.v_path = self.filepath_1.toPlainText()
        self.ic_path = self.filepath_2.toPlainText()
        self.data_format = self.comboBox.currentText()
        self.model_select = self.comboBox_2.currentText()
        self.window_size = self.plainTextEdit_9.toPlainText()
        self.epoch_num = self.plainTextEdit_10.toPlainText()
        self.batch_size = self.plainTextEdit_11.toPlainText()
        self.train_ratio = self.plainTextEdit_12.toPlainText()
        self.best_model_path = self.filepath_3.toPlainText()
        self.best_model_name = self.filepath_5.toPlainText()
        self.parameter_path = self.filepath_4.toPlainText()
        variables = [
            self.v_path,
            self.ic_path,
            self.data_format,
            self.window_size,
            self.epoch_num,
            self.batch_size,
            self.train_ratio,
            self.best_model_path,
            self.best_model_name,
            self.parameter_path
        ]
        all_variables_non_empty = all(variable for variable in variables)
        # self.status_label.setText("开始训练模型,当前选择的模型{}".format(self.model_select))
        if all_variables_non_empty:
            try:
            # self.status_label.setText("开始训练模型,当前选择的模型{}".format(self.model_select))
                self.window_size = int(self.plainTextEdit_9.toPlainText())
                self.epoch_num = int(self.plainTextEdit_10.toPlainText())
                self.batch_size = int(self.plainTextEdit_11.toPlainText())
                self.train_ratio = float(self.plainTextEdit_12.toPlainText())
                total_loss, total_vaildloss = soh_model.train_model_wrapper(self.model_select, self.best_model_path,
                                                                            self.best_model_name,
                                                                            self.parameter_path,
                                                                            self.window_size, self.epoch_num,
                                                                            self.batch_size, self.train_ratio,
                                                                            self.data_format,
                                                                            self.v_data, self.ic_data, self.output_size,
                                                                            self.nomi_capa.text(), self.eol.text())
                self.icloss.clf()
                self.canvas_loss.draw()
                Loss = self.icloss.add_subplot(111)
                Loss.plot(range(1, len(total_loss) + 1), total_loss, 'bo', label='trainloss')
                Loss.plot(range(1, len(total_vaildloss) + 1), total_vaildloss, 'r', label='validloss')
                Loss.set_title('loss_figure')
                Loss.set_ylabel('loss')
                Loss.set_xlabel('epoch_num')
                Loss.legend()
                self.canvas_loss.draw()
                self.rename(self.best_model_path, self.best_model_name)

                # self.status_label.setText(
                #     "模型训练完成，模型类型{}，模型路径{}，模型名称{}.pth, 超参数路径{}_parameter.txt".format(
                #         self.model_select, self.best_model_path, self.best_model_name, self.best_model_path))
            except Exception as e:
                # 捕获异常并显示错误消息
                QMessageBox.critical(self, '出错啦', str(e))

        else:
            self.lackdata()
            # self.status_label.setText("请检查输入数据")

    def test_model(self):
        self.v_path = self.filepath_1.toPlainText()
        self.ic_path = self.filepath_2.toPlainText()
        self.data_format = self.comboBox.currentText()
        self.model_set = self.comboBox_2.currentText()
        self.model_import_path = self.filepath_6.toPlainText()
        self.param_import_path = self.filepath_7.toPlainText()
        variables = [
            self.v_path,
            self.ic_path,
            self.data_format,
            self.param_import_path,
            self.model_import_path
        ]
        if all(self.v_path and self.ic_path and self.model_import_path and self.param_import_path and not variable.empty
               for variable in [self.v_data, self.ic_data]):
            try:
                # self.status_label.setText("开始预测")
                results, MAE, RMSE, R2, num_classes, ground, predict = soh_model.test_model_wrapper(
                    self.model_import_path,
                    self.param_import_path,
                    self.data_format,
                    self.v_data,
                    self.ic_data)

                '''绘图区'''

                MAE_worst = np.argmax(MAE)
                MAE_best = np.argmin(MAE)
                v_index = np.linspace(2.01, 3.60, num_classes)

                self.groundvspred.clf()
                self.truevspred.clf()
                self.canva_gvp.draw()
                self.canva_tvp.draw()

                gvt = self.groundvspred.add_subplot(111)
                tve = self.truevspred.add_subplot(111)

                # 直接绘图预测值和实际值
                cycle_index = range(len(ground))
                ground_color = np.array([171, 58, 41]) / 255
                predict_color = np.array([19, 103, 158]) / 255
                gvt.plot(cycle_index, ground, color=ground_color, label='ground', linewidth=3)
                gvt.scatter(cycle_index, predict, color=predict_color, label='pred', s=15)
                point_x = cycle_index[int(len(cycle_index) / 2)]
                point_y = predict[int(len(predict) / 2)]
                # gvt.annotate(f'R²: {R2:.2f}',
                #              xy=(point_x, point_y),
                #              xytext=(point_x - 20, point_y - 20),
                #              textcoords='data',
                #              fontsize=12,  # 可以调整字体大小
                #              arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=.2'))
                gvt.set_title('ground vs predict')
                gvt.set_ylabel('SOH (%)')
                gvt.set_xlabel('cycle index')
                gvt.legend()
                self.canva_gvp.draw()

                # 映射图
                color1 = np.array([98, 190, 166]) / 255
                tve.scatter(ground, predict, color=color1, s=70)  # 映射点
                # 计算拟合直线（线性回归）
                coefficients = np.polyfit(ground.ravel(), predict.ravel(), 1)
                polynomial = np.poly1d(coefficients)
                y_fit = polynomial(ground)
                # 绘制拟合直线
                tve.plot(ground, y_fit, color=color1, linewidth=3)  # 拟合直线通常用红色显示
                lims = [
                    min(min(ground), min(predict)),
                    max(max(ground), max(predict)),
                ]
                color2 = np.array([231, 98, 84]) / 255
                tve.plot(lims, lims, alpha=0.75, zorder=0, color=color2, linewidth=3)  # 45度线
                tve.set_xlim(lims)
                tve.set_ylim(lims)
                tve.set_title('True vs Estimate SOH')
                tve.set_ylabel('Estimate SOH (%)')
                tve.set_xlabel('True SOH (%)')
                self.canva_tvp.draw()

                a = str(RMSE)
                self.label_RMSE.setText(a)
                a = str(MAE)
                self.label_MAE.setText(a)
                a = str(R2)
                self.label_R.setText(a)
                # self.mae_rmse.clf()
                # self.mae_rmse1.clf()
                # self.mae_rmse2.clf()
                # self.mae_rmse3.clf()
                # self.canva_rmse_best.draw()
                # self.canva_mae_worst.draw()
                # self.canva_rmse_worst.draw()
                # self.canva_mae_best.draw()
                # # MAE_worst
                # mae_worst = self.mae_rmse1.add_subplot(111)
                #
                # mae_worst.plot(v_index, ground[MAE_worst, :], 'bo', label='ground')
                # mae_worst.plot(v_index, predict[MAE_worst, :], 'r', label='pred')
                # mae_worst.set_title('MAE_worst')
                # mae_worst.set_ylabel('Incremental Capacity(Ah/V)')
                # mae_worst.set_xlabel('voltage (V)')
                # mae_worst.legend()
                # # self.mae_rmse.subplots_adjust(left=0.3, right=0.4, top=0.2, bottom=0.1)
                # self.canva_mae_worst.draw()
                # # mae_worst.cla()
                #
                # # MAE_best
                # mae_best = self.mae_rmse.add_subplot(111)
                # mae_best.plot(v_index, ground[MAE_best, :], 'bo', label='ground')
                # mae_best.plot(v_index, predict[MAE_best, :], 'r', label='pred')
                # mae_best.set_title('MAE_best')
                # mae_best.set_ylabel('Incremental Capacity(Ah/V)')
                # mae_best.set_xlabel('voltage (V)')
                # mae_best.legend()
                # self.canva_mae_best.draw()
                # # mae_best.cla()
                #
                # # RMSE_figure
                # RMSE_worst = np.argmax(RMSE)
                # RMSE_best = np.argmin(RMSE)
                # v_index = np.linspace(2.01, 3.60, num_classes)
                #
                # # RMSE_worst
                # rmse_worst = self.mae_rmse3.add_subplot(111)
                # rmse_worst.plot(v_index, ground[RMSE_worst, :], 'bo', label='ground')
                # rmse_worst.plot(v_index, predict[RMSE_worst, :], 'r', label='pred')
                # rmse_worst.set_title('RMSE_worst')
                # rmse_worst.set_ylabel('Incremental Capacity(Ah/V)')
                # rmse_worst.set_xlabel('voltage (V)')
                # rmse_worst.legend()
                # self.canva_rmse_worst.draw()
                # # rmse_worst.cla()
                #
                # # RMSE_best
                # rmse_best = self.mae_rmse2.add_subplot(111)
                # rmse_best.plot(v_index, ground[RMSE_best, :], 'bo', label='ground')
                # rmse_best.plot(v_index, predict[RMSE_best, :], 'r', label='pred')
                # rmse_best.set_title('RMSE_best')
                # rmse_best.set_ylabel('Incremental Capacity(Ah/V)')
                # rmse_best.set_xlabel('voltage (V)')
                # rmse_best.legend()
                # self.canva_rmse_best.draw()
                # rmse_best.cla()
                # self.status_label.setText("绘图完成")

            except Exception as e:
                # 捕获异常并显示错误消息
                QMessageBox.critical(self, '出错啦', str(e))
        else:
            self.lackdata()
            # self.status_label.setText("请检查输入数据")

    def predict_model(self):
        self.v_path = self.filepath_1.toPlainText()
        self.data_format = self.comboBox.currentText()
        self.model_import_path = self.filepath_6.toPlainText()
        self.param_import_path = self.filepath_7.toPlainText()
        self.data_store_path = self.filepath_8.toPlainText()
        variables = [
            self.v_path,
            self.data_format,
            self.param_import_path,
            self.model_import_path,
            self.data_store_path

        ]
        all_variables_non_empty = all(variable for variable in variables)
        if all_variables_non_empty:
            try:
                v_data, plot_cycle, cycle_num = soh_getin.soh_getin_predict(self.data_format, self.v_path)
                # 绘图
                ##input_figure,Q_V的图像
                self.icindown.clf()
                self.icinup.clf()
                self.canvas.draw()
                self.canvas_1.draw()
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

                soh_model.predict_model_wrapper(self.model_import_path, self.param_import_path, self.data_format,
                                                v_data,
                                                self.data_store_path)
                # self.status_label.setText("已成功预测,数据保存在{}目录下".format(self.data_store_path))
                self.finish()
            except Exception as e:
                # 捕获异常并显示错误消息
                QMessageBox.critical(self, '出错啦', str(e))

        else:
            self.lackdata()
            # self.status_label.setText("请检查输入的数据")

    def rename(self, best_model_path, best_model_name):
        files = [f for f in os.listdir(best_model_path) if f.endswith('.pth')]
        if files:
            # 使用提取函数来选择最佳文件
            best_pth_file = max(files, key=lambda x: extract_number(x))
            # best_pth_file = max(files, key=lambda x: int(x[:4].split('_')[0]))
            os.rename(os.path.join(best_model_path, best_pth_file),
                      os.path.join(best_model_path, best_model_name + '.pth'))

            print("File renamed successfully.")
        else:

            print("No .pth files found in the directory.")

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

    def clean(self):
        print('Cleaning up...')
        self.icindown.clf()
        self.icinup.clf()
        self.groundvspred.clf()
        self.truevspred.clf()
        self.canva_gvp.draw()
        self.canva_tvp.draw()
        # self.mae_rmse.clf()
        # self.mae_rmse1.clf()
        # self.mae_rmse2.clf()
        # self.mae_rmse3.clf()
        self.icloss.clf()
        self.canvas_loss.draw()
        # self.canva_rmse_best.draw()
        # self.canva_mae_worst.draw()
        # self.canva_rmse_worst.draw()
        # self.canva_mae_best.draw()
        self.canvas.draw()
        self.canvas_1.draw()

    def finish(self):
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
        msg_box.setWindowTitle('好耶ヽ(✿ﾟ▽ﾟ)ノ')
        msg_box.setText('预测已完成~（此对话框会自动关闭）')
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
