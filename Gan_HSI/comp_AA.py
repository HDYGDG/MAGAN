import matplotlib.pyplot as plt
from pylab import *  # ⽀持中⽂

mpl.rcParams['font.sans-serif'] = ['SimHei']
# names = ['1%', '3%', '5%', '7%', '10%', '15%', '20%']
# names = ['1%', '5%', '10%', '15%', '20%']


# patch size 与训练样本比
# IP
# name = 'Indian Pines'
# SVM
# GoogleNet
# HybridSN
# Transformer
# Vit
# TNT
# SSFTT
# MLFF

y_SVM = [46.38, 62.78, 67.01,70.96,74.73]
y_GoogleNet = [42.70, 54.34,79.41,74.48,90.7]
y_HybridSN = [49.38,54.35,79.41,92.69,93.58]
y_Transformer = [52.64,66.56,75.21,83.20,88.52]
y_Vit= [43.68,52.30,63.88,68.44,72.74]
y_TNT = [41.98,44.57,64.30,70.31,75.83]
y_SSFTT = [62.93,79.45,87.06,90.56,95.37]
y_MLFF = [66.17,87.04,90.26,93.59,97.61]
names = ['1%', '3%', '5%', '7%', '10%']
#


# y_SVM = [92.87,94.81,95.35,95.573,95.966]
# y_GoogleNet = [93.03,95.66,95.95,99.17,97.66]
# y_HybridSN = [92.58,99.37,99.82,99.78,99.89]
# y_Transformer = [96.57,99.32,99.83,99.83,99.95]
# y_Vit= [96.83,99.03,99.03,99.64,99.76]
# y_TNT = [90.78,99.32,99.71,99.78,99.85]
# y_SSFTT = [98.95,99.55,99.726,99.7,99.97]
# y_MLFF = [99.01,99.67,99.77,99.77,99.98]
# names = ['1%', '3%', '5%', '7%', '10%']
#
# # # SA
# name = 'Salinas'


# y_SVM = [82.33,90.12,90.65,91.78,92.51]
# y_GoogleNet = [91.01,90.987,92.34,98.07,98.07]
# y_HybridSN = [83.15,94.67,98.44	,98.53,99.05]
# y_Transformer = [82.88,92.85,93.9,98.21,98.68]
# y_Vit= [82.10 ,91.4,94.7,95.79,96.45]
# y_TNT = [80.98,92.41,94.44,95.57,97.19]
# y_SSFTT = [93.71,96.03,98.78,99.32,99.24]
# y_MLFF = [95.01,97.78,99.42,99.596,99.66]
# names = ['1%', '3%', '5%', '7%', '10%']

# y13 = [99.91, 99.90, 99.91]
# y15 = [99.95, 99.96, 99.98]
# y17 = [99.93, 99.93, 99.96]
# y19 = [99.89, 99.96, 99.94]
# y21 = [99.92, 99.94, 99.96]
# y23 = [99.93, 99.97, 99.96]
# names = ['10%', '15%', '20%']

# PU
name = 'Pavia University'


# y13 = [97.10, 98.81, 99.15, 99.36]
# y15 = [97.15, 98.88, 99.39, 99.67]
# y17 = [97.25, 98.92, 99.67, 99.63]
# y19 = [97.36, 98.99, 99.39, 99.21]
# y21 = [97.55, 99.10, 99.70, 99.79]
# y23 = [97.66, 99.27, 99.82, 99.89]
# names = ['1%', '3%', '5%', '7%']
#
# y13 = [99.51, 99.78, 99.86]
# y15 = [99.65, 99.82, 99.86]
# y17 = [99.69, 99.38, 99.86]
# y19 = [99.80, 99.76, 99.94]
# y21 = [99.71, 99.89, 99.91]
# y23 = [99.82, 99.93, 99.95]
# names = ['10%', '15%', '20%']


# plt.plot(x, y, 'ro-')
# plt.plot(x, y1, 'bo-')
# pl.xlim(-1, 11) # 限定横轴的范围
# pl.ylim(-1, 110) # 限定纵轴的范围


x = range(len(names))
mssize = 10
plt.plot(x, y_SVM, marker='o', mec='r', ms=mssize, label=u'SVM')
plt.plot(x, y_GoogleNet, marker='*', mec='b', ms=mssize, label=u'GoogleNet')
plt.plot(x, y_HybridSN, marker='D', mec='y', ms=mssize, label=u'HybridSN')
plt.plot(x, y_Transformer, marker='X', mec='c', ms=mssize, label=u'Transformer')
plt.plot(x, y_Vit, marker='s', mec='g', ms=mssize, label=u'Vit')
plt.plot(x, y_TNT, marker='p', mec='w', ms=mssize, label=u'TNT')
plt.plot(x, y_SSFTT, marker='h', mec='k', ms=mssize, label=u'SSFTT')
plt.plot(x, y_MLFF, marker='v', mec='m', ms=mssize, label=u'MLFF')
plt.rcParams.update({'font.size': 9})
plt.legend()  # 让图例⽣效
plt.xticks(x, names, rotation=45, size=12)
plt.yticks(fontsize=10)
# 12
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel("Train sample ratio(%)", fontsize=15)  # X轴标签
plt.ylabel("AA(%)", fontsize=18)  # Y轴标签
# plt.title("{0} Classification Overall accuracies".format(name))  # 标题
# plt.title("{0}".format(name), fontsize=20)  # 标题
plt.show()
