#老化机制分析
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#界面选择的变量只有四个data_formate,ic_path,download_path,origin_ic
#接口代码
def LAM_analysis(data_formate,ic_path,download_path,origin_ic):
    #数据载入并绘制数据图
    if data_formate =='Txt':
        ic_data = np.genfromtxt(ic_path, delimiter=',')
        ic_data = np.delete(ic_data, -1, axis=1)  # ic数值为单位C/V或者mAh/V
        #绘图
        cycle_num = len(ic_data)
        plot_cycle = range(0, cycle_num, int(cycle_num / 10))
        download_name = download_path + "/analysis_result.txt"
        # fig, ax = plt.subplots()
        # for cycle in plot_cycle:
        #     plt.plot(range(len(ic_data[cycle])), ic_data[cycle]/3600, color=plt.cm.viridis(cycle / cycle_num))
        # plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, cycle_num)), ax=ax,label='Cycles', ticks=plot_cycle)
        # plt.title('IC data preview')
        # plt.xlabel('Point index')
        # plt.ylabel('Incremental Capacity (Ah/V)')
        # plt.show()
    elif data_formate == 'Excel':
        ic_data = pd.read_excel(ic_path)
        cycle_num = len(ic_data)
        plot_cycle = range(0, cycle_num, int(cycle_num / 10))
        # fig, ax = plt.subplots()
        # for cycle in plot_cycle:
        #     plot_ic=ic_data.iloc[cycle,:]
        #     plt.plot(range(len(plot_ic)), plot_ic / 3600, color=plt.cm.viridis(cycle / cycle_num))
        # plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, cycle_num)),ax=ax, label='Cycles',
        #                 ticks=plot_cycle)
        # plt.title('IC data preview')
        # plt.xlabel('Point index')
        # plt.ylabel('Incremental Capacity (Ah/V)')
        # plt.show()
        download_name = download_path + "/analysis_result.xlsx"

    # download_name=download_path+"/analysis_result."+data_formate
    max_ic = np.min(ic_data, axis=1)
    if (min(max_ic)<-20):
        max_ic=max_ic/3600
    LAM=(origin_ic-max_ic)/origin_ic*100
    #绘制量化结果图
    # plt.figure(dpi=150)
    # color=np.array([231,98,84])/255
    # plt.plot(range(1,len(LAM)+1), LAM, color=color)
    # plt.title('LAM quantization result')
    # plt.ylabel('LAM (%)')
    # plt.xlabel('cycle index')
    # plt.legend()
    # plt.show()

    if data_formate == 'Txt':
        np.savetxt(download_name, LAM, delimiter=",")
    elif data_formate == 'Excel':
        pd.DataFrame(LAM).to_excel(download_name, index=False, header=False)

    return ic_data, cycle_num, plot_cycle, LAM

# data_formate="txt"
# ic_path='D:/Desktop/电气电子/method_code/6C-60per_3C/CH30/ic.txt'#这个是IC数据路径
# download_path='D:/Desktop'#这个是存储路径
# origin_ic=-8#这个是输入的原始ic值，用户自己输入
# LAM_analysis(data_formate,ic_path,download_path,origin_ic)

def LAM_analysis_wrap(formate,ic_path,download_path,origin_ic):
    ic_data, cycle_num, plot_cycle, LAM = LAM_analysis(formate, ic_path, download_path, origin_ic)

    return ic_data, cycle_num, plot_cycle, LAM

