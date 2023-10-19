import matplotlib.pyplot as plt
from pylab import *  # ⽀持中⽂

mpl.rcParams['font.sans-serif'] = ['SimHei']
names = ['1%', '3%', '5%', '7%', '10%']
x = range(len(names))

# patch size 与训练样本比
# IP
# name= 'Indian Pines'
# y13 = [65.37, 75.19, 82.23, 89.46, 94.23]
# y15 = [67.44, 78.22, 85.14, 91.44, 95.52]
# y17 = [70.14, 79.32, 87.56, 92.64, 96.68]
# y19 = [72.33, 80.16, 90.79, 94.16, 97.21]
# y21 = [71.34, 78.66, 87.41, 92.43, 96.41]

# SA
# name = 'Pavia University'
# y13 = [89.46,90.34,96.03,96.66,97.18]
# y15 = [90.17,90.87,96.53,96.97,97.74]
# y17 = [90.34,91.76,96.98,97.21,98.29]
# y19 = [91.65,92.14,97.26,97.42,98.67]
# y21 = [90.22,91.75,97.04,97.29,98.01]

# PU
# name='Salinas'
y13 = [97.07, 97.84, 98.12, 98.26, 99.17]
y15 = [97.54, 98.09, 98.29, 98.49, 99.25]
y17 = [97.96, 98.24, 98.45, 98.94, 99.38]
y19 = [98.32, 98.62, 98.84, 99.32, 99.57]
y21 = [97.89, 98.19, 98.40, 98.90, 99.29]

# plt.plot(x, y, 'ro-')
# plt.plot(x, y1, 'bo-')
# pl.xlim(-1, 11) # 限定横轴的范围
# pl.ylim(-1, 110) # 限定纵轴的范围
plt.plot(x, y13, marker='o', mec='r', mfc='w', label=u'1:9')
plt.plot(x, y15, marker='*', mec='b', ms=5, label=u'2:8')
plt.plot(x, y17, marker='D', mec='y', ms=5, label=u'3:7')
plt.plot(x, y19, marker='X', mec='c', ms=5, label=u'4:6')
plt.plot(x, y21, marker='s', mec='g', ms=5, label=u'5:5')

plt.legend()  # 让图例⽣效
plt.xticks(x, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.14)
plt.xlabel("Train sample ratio(%)")  # X轴标签
plt.ylabel("OA(%)")  # Y轴标签
# plt.title("{0} Classification Overall accuracies".format(name))  # 标题
plt.show()
