import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 设置主窗口标题和尺寸
        self.setWindowTitle("Matplotlib in PyQt")
        self.setGeometry(100, 100, 800, 600)

        # 创建主widget和布局
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 创建Matplotlib图形左侧
        self.figure_left = Figure(figsize=(5, 4), dpi=100)
        self.canvas_left = FigureCanvas(self.figure_left)
        layout.addWidget(self.canvas_left)

        # 创建Matplotlib图形右侧
        self.figure_right = Figure(figsize=(5, 4), dpi=100)
        self.canvas_right = FigureCanvas(self.figure_right)
        layout.addWidget(self.canvas_right)

        # 创建按钮并连接槽函数
        button_left = QPushButton("Plot Left")
        button_left.clicked.connect(self.plot_left)
        layout.addWidget(button_left)

        button_right = QPushButton("Plot Right")
        button_right.clicked.connect(self.plot_right)
        layout.addWidget(button_right)

    def plot_left(self):
        # 绘制左侧图形
        ax = self.figure_left.add_subplot(111)
        ax.plot([1, 2, 3, 4], [10, 20, 25, 30], '-o')
        ax.set_title('Left Plot')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        self.canvas_left.draw()

    def plot_right(self):
        # 绘制右侧图形
        ax = self.figure_right.add_subplot(111)
        ax.plot([1, 2, 3, 4], [30, 25, 20, 10], '-o')
        ax.set_title('Right Plot')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        self.canvas_right.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
