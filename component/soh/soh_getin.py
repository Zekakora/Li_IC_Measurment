#数据加载版块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
#这两个是数据加载的两个路径,尾缀是txt还是xlsx看用户选择的数据格式
#用户可能直接从windows文件夹中复制路径，所以要注意路径的斜杠，是否可以在前面加r控制只读字符串
# data_formate='txt'#用户选择的数据格式
# v_path='D:/Desktop/deal_data/6C-60per_3C/CH29/v.txt'#这个是Q_V数据路径
# capacity_path='D:/Desktop/deal_data/6C-60per_3C/CH29/capacity.txt'#这个是IC数据路径
# nominal_capacity=1.1#这个是电池的标称容量
# eol=nominal_capacity*80/100#80才是用户输入的数字，输入后我们需要将其转变为百分比小数
#soh_getin_predict用于加载训练数据输入
def soh_getin (data_formate,v_path,capacity_path,nominal_capacity,eol):
    #包含两个数据的预览图
    if data_formate =='TXT':
        capacity_data = np.genfromtxt(capacity_path, delimiter=',')
        capacity_data = np.delete(capacity_data, -1, axis=1)  # ic数值为单位C/V或者mAh/V
        soh_data = (capacity_data-eol) / (nominal_capacity-eol)*100
        output_size= soh_data.shape[1]
        v_data = []
        with open(v_path, 'r') as file:
            v_data = file.readlines()
        cycle_num = len(v_data)
        plot_cycle = range(0, cycle_num, int(cycle_num / 10))
        ##input_figure,Q_V的图像
        #plt.figure()
        figure, ax = plt.subplots()
        for cycle in plot_cycle:
            plot_y = v_data[cycle].split(',')
            plot_y = np.array(plot_y)  # 假设以逗号为分隔符
            plot_y = np.delete(plot_y, -1, axis=0)
            plot_y = plot_y.astype(float)
            plt.plot(range(len(plot_y)), plot_y, color=plt.cm.seismic(cycle / cycle_num))
        plt.colorbar(plt.cm.ScalarMappable(cmap='seismic',norm=plt.Normalize(0, cycle_num)),
                     ax=ax,label='Cycles',ticks=plot_cycle)
        plt.title('Q-V data preview')
        plt.xlabel('Point index')
        plt.ylabel('Voltage (V)')
        plt.show()

        ##Output_figure,soh的图像
        # 设置颜色相关参数
        fig, ax = plt.subplots()
        norm = Normalize(vmin=soh_data.min(), vmax=soh_data.max())
        cmap = plt.get_cmap('seismic')
        # 创建ScalarMappable对象
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # 只是为了满足colorbar的需要
        # 绘制线，颜色随soh值变化
        for i in range(cycle_num - 1):
            plt.plot(range(1,cycle_num+1)[i:i + 2], soh_data[i:i + 2], color=sm.to_rgba(soh_data[i]))
        # 添加colorbar
        cbar = plt.colorbar(sm, ax=ax,orientation='vertical')
        cbar.set_label('SOH value')
        plt.title('SOH data preview')
        plt.xlabel('cycle index')
        plt.ylabel('SOH (%)')
        # 显示图表
        plt.show()

    elif data_formate == 'Excel':
        capacity_data = pd.read_excel(capacity_path)
        soh_data = (capacity_data - eol) / (nominal_capacity - eol) * 100
        output_size = soh_data.shape[1]

        v_data=pd.read_excel(v_path)
        cycle_num = len(v_data)
        plot_cycle = range(0, cycle_num, int(cycle_num / 10))
        ##input_figure,Q_V的图像
        fig, ax = plt.subplots()
        for cycle in plot_cycle:
            plot_v=v_data.iloc[cycle,:]
            plot_v = [value for value in plot_v if pd.notna(value)]
            plt.plot(range(len(plot_v)), plot_v, color=plt.cm.seismic(cycle / cycle_num))
        plt.colorbar(plt.cm.ScalarMappable(cmap='seismic', norm=plt.Normalize(0, cycle_num)),
                     ax=ax,label='Cycles',ticks=plot_cycle)
        plt.title('Q-V data preview')
        plt.xlabel('Point index')
        plt.ylabel('Voltage (V)')
        plt.show()

        ##Output_figure,soh的图像
        # 设置颜色相关参数
        fig, ax = plt.subplots()
        norm = Normalize(vmin=soh_data.min(), vmax=soh_data.max())
        cmap = plt.get_cmap('seismic')
        # 创建ScalarMappable对象
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # 只是为了满足colorbar的需要
        # 绘制线，颜色随soh值变化
        for i in range(cycle_num - 1):
            plt.plot(range(1, cycle_num + 1)[i:i + 2], soh_data.iloc[i:i + 2,0], color=sm.to_rgba(soh_data.iloc[i,0]))
        # 添加colorbar
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('SOH value')
        plt.title('SOH data preview')
        plt.xlabel('cycle index')
        plt.ylabel('SOH (%)')
        # 显示图表
        plt.show()
    return v_data,soh_data,output_size
# v_data,soh_data,output_size=soh_getin(data_formate,v_path,capacity_path,nominal_capacity,eol)


#soh_getin_predict用于加载预测数据输入
def soh_getin_predict (data_formate,v_path):
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
        #     plt.plot(range(len(plot_y)), plot_y, color=plt.cm.seismic(cycle / cycle_num))
        # plt.colorbar(plt.cm.ScalarMappable(cmap='seismic',norm=plt.Normalize(0, cycle_num)),label='Cycles',ticks=plot_cycle)
        # plt.title('Q-V data preview')
        # plt.xlabel('Point index')
        # plt.ylabel('Voltage (V)')
        # plt.show()
    elif data_formate == 'Excel':
        v_data=pd.read_excel(v_path)
        cycle_num = len(v_data)
        plot_cycle = range(0, cycle_num, int(cycle_num / 10))
        ##input_figure,Q_V的图像
        # plt.figure()
        # for cycle in plot_cycle:
        #     plot_v=v_data.iloc[cycle,:]
        #     plot_v = [value for value in plot_v if pd.notna(value)]
        #     plt.plot(range(len(plot_v)), plot_v, color=plt.cm.seismic(cycle / cycle_num))
        # plt.colorbar(plt.cm.ScalarMappable(cmap='seismic', norm=plt.Normalize(0, cycle_num)), label='Cycles',ticks=plot_cycle)
        # plt.title('Q-V data preview')
        # plt.xlabel('Point index')
        # plt.ylabel('Voltage (V)')
        # plt.show()
    return v_data,plot_cycle, cycle_num






