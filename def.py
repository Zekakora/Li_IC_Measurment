def set_text(self, fname):
    self.textBrowser.clear()
    fname = str(fname)
    self.textBrowser.append(fname)
    # self.textBrowser.append(fname)


def choosefile(self):
    fname, _ = QFileDialog.getOpenFileName(None, '选择文件', '/home')
    if fname:  # 如果用户选择了文件
        self.set_text(fname)
    self.evaluation()


def evaluation(self):
    max = 5
    min = 2
    ave = 3.5
    self.textBrowser_1.clear()
    result = eval_conbime(max, ave, min)
    self.textBrowser_1.append(result)


def eval_conbime(max, ave, min):
    return f"最大：{max}\t\t平均：{ave}\t\t最小：{min}"
