#数据加载版块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#这两个是数据加载的两个路径,尾缀是txt还是xlsx看用户选择的数据格式
#用户可能直接从windows文件夹中复制路径，所以要注意路径的斜杠，是否可以在前面加r控制只读字符串
data_formate='txt'#用户选择的数据格式
v_path='D:/Desktop/deal_data/6C-60per_3C/CH29/v.txt'#这个是Q_V数据路径
ic_path='D:/Desktop/deal_data/6C-60per_3C/CH29/ic.txt'#这个是IC数据路径

def ic_getin (data_formate,v_path,ic_path):
    #包含两个数据的预览图
    if data_formate =='txt':
        ic_data = np.genfromtxt(ic_path, delimiter=',')
        ic_data = np.delete(ic_data, -1, axis=1)  # ic数值为单位C/V或者mAh/V
        output_size= ic_data.shape[1]
        v_data = []
        with open(v_path, 'r') as file:
            v_data = file.readlines()
        cycle_num = len(v_data)
        plot_cycle = range(0, cycle_num, int(cycle_num / 10))

        ##input_figure,Q_V的图像
        plt.figure()
        for cycle in plot_cycle:
            plot_y = v_data[cycle].split(',')
            plot_y = np.array(plot_y)  # 假设以逗号为分隔符
            plot_y = np.delete(plot_y, -1, axis=0)
            plot_y = plot_y.astype(np.float)
            plt.plot(range(len(plot_y)), plot_y, color=plt.cm.viridis(cycle / cycle_num))
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis',norm=plt.Normalize(0, cycle_num)),label='Cycles',ticks=plot_cycle)
        plt.title('Q-V data preview')
        plt.xlabel('Point index')
        plt.ylabel('Voltage (V)')
        plt.show()

        ##Output_figure,IC的图像
        plt.figure()
        for cycle in plot_cycle:
            plt.plot(range(len(ic_data[cycle])), ic_data[cycle]/3600, color=plt.cm.viridis(cycle / cycle_num))
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, cycle_num)), label='Cycles', ticks=plot_cycle)
        plt.title('IC data preview')
        plt.xlabel('Point index')
        plt.ylabel('Incremental Capacity (Ah/V)')
        plt.show()

    elif data_formate == 'excel':
        ic_data = pd.read_excel(ic_path)
        output_size = ic_data.shape[1]
        v_data=pd.read_excel(v_path)
        cycle_num = len(v_data)
        plot_cycle = range(0, cycle_num, int(cycle_num / 10))
        ##input_figure,Q_V的图像
        plt.figure()
        for cycle in plot_cycle:
            plot_v=v_data.iloc[cycle,:]
            plot_v = [value for value in plot_v if pd.notna(value)]
            plt.plot(range(len(plot_v)), plot_v, color=plt.cm.viridis(cycle / cycle_num))
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, cycle_num)), label='Cycles',ticks=plot_cycle)
        plt.title('Q-V data preview')
        plt.xlabel('Point index')
        plt.ylabel('Voltage (V)')
        plt.show()

        ##Output_figure,IC的图像
        plt.figure()
        for cycle in plot_cycle:
            plot_ic=ic_data.iloc[cycle,:]
            plt.plot(range(len(plot_ic)), plot_ic / 3600, color=plt.cm.viridis(cycle / cycle_num))
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, cycle_num)), label='Cycles',
                     ticks=plot_cycle)
        plt.title('IC data preview')
        plt.xlabel('Point index')
        plt.ylabel('Incremental Capacity (Ah/V)')
        plt.show()
    return v_data,ic_data,output_size
#v_data,ic_data,output_size=ic_getin(data_formate,v_path,ic_path)
# print(v_data)


