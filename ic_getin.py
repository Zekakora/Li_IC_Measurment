#数据加载版块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
#这两个是数据加载的两个路径,尾缀是txt还是xlsx看用户选择的数据格式
#用户可能直接从windows文件夹中复制路径，所以要注意路径的斜杠，是否可以在前面加r控制只读字符串
# data_formate='excel'#用户选择的数据格式
# v_path='D:/v.xlsx'#这个是Q_V数据路径
# ic_path='D:/ic.xlsx'#这个是IC数据路径
#ic_getin_predict用于加载训练数据输入


def ic_getin (data_formate,v_path,ic_path):
    #包含两个数据的预览图
    if data_formate =='TXT':
        ic_data = np.genfromtxt(ic_path, delimiter=',')
        ic_data = np.delete(ic_data, -1, axis=1)  # ic数值为单位C/V或者mAh/V
        output_size= ic_data.shape[1]
        v_data = []
        with open(v_path, 'r') as file:
            v_data = file.readlines()
        cycle_num = len(v_data)
        plot_cycle = range(0, cycle_num, int(cycle_num / 10))
        fig, ax = plt.subplots()
        ##input_figure,Q_V的图像
        #plt.figure()
        for cycle in plot_cycle:
            plot_y = v_data[cycle].split(',')
            plot_y = np.array(plot_y)  # 假设以逗号为分隔符
            plot_y = np.delete(plot_y, -1, axis=0)
            plot_y = plot_y.astype(float)
            plt.plot(range(len(plot_y)), plot_y, color=plt.cm.viridis(cycle / cycle_num))
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, cycle_num)),ax=ax,label='Cycles',ticks=plot_cycle)
        plt.title('Q-V data preview')
        plt.xlabel('Point index')
        plt.ylabel('Voltage (V)')
        plt.show()

        ##Output_figure,IC的图像
        fig, ax = plt.subplots()
        for cycle in plot_cycle:
            plt.plot(range(len(ic_data[cycle])), ic_data[cycle]/3600, color=plt.cm.viridis(cycle / cycle_num))
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, cycle_num)), ax=ax,label='Cycles', ticks=plot_cycle)
        plt.title('IC data preview')
        plt.xlabel('Point index')
        plt.ylabel('Incremental Capacity (Ah/V)')
        plt.show()
        # 判断老化机制,使用一个输出框判断
        peak_data = ic_data[0]
        peaks, _ = find_peaks(abs(peak_data), height=-6 * 3600, threshold=0.1 * 3600)
        peaks_num = len(peaks)
        if peaks_num == 1:
            peak_data = ic_data[-1]
            peaks1, _ = find_peaks(abs(peak_data), height=-6 * 3600, threshold=0.1 * 3600)
            if peaks1[0] < peaks[0]:
                degration='活性物质损失'
            else:
                degration='锂库存和活性物质损失'
        elif peaks_num == 2:
            degration='锂库存和活性物质损失'

    elif data_formate == 'Excel':
        ic_data = pd.read_excel(ic_path)
        output_size = ic_data.shape[1]
        v_data=pd.read_excel(v_path)
        cycle_num = len(v_data)
        plot_cycle = range(0, cycle_num, int(cycle_num / 10))
        ##input_figure,Q_V的图像
        fig, ax = plt.subplots()
        for cycle in plot_cycle:
            plot_v=v_data.iloc[cycle,:]
            plot_v = [value for value in plot_v if pd.notna(value)]
            plt.plot(range(len(plot_v)), plot_v, color=plt.cm.viridis(cycle / cycle_num))
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, cycle_num)),ax=ax, label='Cycles',ticks=plot_cycle)
        plt.title('Q-V data preview')
        plt.xlabel('Point index')
        plt.ylabel('Voltage (V)')
        plt.show()

        ##Output_figure,IC的图像
        fig, ax = plt.subplots()
        for cycle in plot_cycle:
            plot_ic=ic_data.iloc[cycle,:]
            plt.plot(range(len(plot_ic)), plot_ic / 3600, color=plt.cm.viridis(cycle / cycle_num))
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, cycle_num)),ax=ax, label='Cycles',
                     ticks=plot_cycle)
        plt.title('IC data preview')
        plt.xlabel('Point index')
        plt.ylabel('Incremental Capacity (Ah/V)')
        # plt.show()
        # 判断老化机制,使用一个输出框判断
        peak_data = ic_data.iloc[0,:]
        peaks, _ = find_peaks(abs(peak_data), height=-6 * 3600, threshold=0.1 * 3600)
        peaks_num = len(peaks)
        if peaks_num == 1:
            peak_data = ic_data.iloc[-1, :]
            peaks1, _ = find_peaks(abs(peak_data), height=-6 * 3600, threshold=0.1 * 3600)
            if peaks1[0] <peaks[0]:
                degration ='活性物质损失'
            else:
                degration='锂库存和活性物质损失'
        elif peaks_num == 2:
            degration='锂库存和活性物质损失'
    return v_data,ic_data,output_size,degration
# v_data,ic_data,output_size,degration=ic_getin(data_formate,v_path,ic_path)
# rows = pd.eval(v_data[0])

#ic_getin_predict用于加载预测数据输入
def ic_getin_predict (data_formate,v_path):
    plot_cycle = None
    v_data = None
    cycle_num = None
    #包含一个数据的预览图
    if data_formate =='TXT':
        v_data = []
        with open(v_path, 'r') as file:
            v_data = file.readlines()
        cycle_num = len(v_data)
        plot_cycle = range(0, cycle_num, int(cycle_num / 10))
        ##input_figure,Q_V的图像
        # plt.figure()
        # for cycle in plot_cycle:
        #     plot_y = v_data[cycle].split(',')
        #     plot_y = np.array(plot_y)  # 假设以逗号为分隔符
        #     plot_y = np.delete(plot_y, -1, axis=0)
        #     plot_y = plot_y.astype(np.float)
        #     plt.plot(range(len(plot_y)), plot_y, color=plt.cm.viridis(cycle / cycle_num))
        # plt.colorbar(plt.cm.ScalarMappable(cmap='viridis',norm=plt.Normalize(0, cycle_num)),label='Cycles',ticks=plot_cycle)
        # plt.title('Q-V data preview')
        # plt.xlabel('Point index')
        # plt.ylabel('Voltage (V)')
        # plt.show()
    elif data_formate == 'Excel':
        v_data=pd.read_excel(v_path)
        cycle_num = len(v_data)
        plot_cycle = range(0, cycle_num, int(cycle_num / 10))
        ##input_figure,Q_V的图像
    #     plt.figure()
    #     for cycle in plot_cycle:
    #         plot_v=v_data.iloc[cycle,:]
    #         plot_v = [value for value in plot_v if pd.notna(value)]
    #         plt.plot(range(len(plot_v)), plot_v, color=plt.cm.viridis(cycle / cycle_num))
    #     plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, cycle_num)), label='Cycles',ticks=plot_cycle)
    #     plt.title('Q-V data preview')
    #     plt.xlabel('Point index')
    #     plt.ylabel('Voltage (V)')
    #     plt.show()
    return v_data,plot_cycle, cycle_num
