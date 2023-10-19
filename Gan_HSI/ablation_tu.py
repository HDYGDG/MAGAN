import matplotlib.pyplot as plt
from pylab import *  # ⽀持中⽂

import numpy as np

mpl.rcParams['font.sans-serif'] = ['SimHei']
# 返回size个0-1的随机数
# IP
name = 'Indian Pines'
y_only_MAE = [55.32,70.43]
y_MAE_G = [62.46,73.42]
y_MAE_3 = [67.16,76.61]
y_GAN = [53.14,72.16]
y_proposed = [72.33,80.16]

# name = 'Salinas'
# y_only_MAE = [87.16,90.08]
# y_MAE_G = [90.41,90.52]
# y_MAE_3 = [90.22,91.84]
# y_GAN = [85.14,90.34]
# y_proposed = [91.65,92.14]
#
# # name = 'Pavia University'
# y_only_MAE = [96.53,97.53]
# y_MAE_G = [97.71,98.2]
# y_MAE_3 = [98.1,98.3]
# y_GAN = [96.41,97.41]
# y_proposed = [98.32,98.62]

total_width, n = 1, 3
width = total_width / n / 2
names = ['1%', '3%']
x = np.arange(len(names))  # x轴标题


plt.bar(x - 2 * width, y_only_MAE, width=width)
plt.bar(x - 1 * width, y_MAE_G, width=width)
plt.bar(x - 0 * width, y_MAE_3, width=width)
plt.bar(x + 1 * width, y_GAN, width=width)
plt.bar(x + 2 * width, y_proposed, width=width)


plt.rcParams.update({'font.size': 10})
plt.xticks(x, names, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Train sample ratio(%)", fontsize=18)  # X轴标签
plt.ylabel("OA(%)", fontsize=20, labelpad=-2)  # Y轴标签
plt.ylim(50, 85)
# plt.title("{0} ".format(name), fontsize=20)  # 标题
# 显示图例
plt.legend()
# 显示柱状图
plt.show()
