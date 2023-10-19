import matplotlib.pyplot as plt
from pylab import *  # ⽀持中⽂

mpl.rcParams['font.sans-serif'] = ['SimHei']
names = ['1%', '3%', '5%', '7%', '10%']
x = range(len(names))

# patch size 与训练样本比
# IP
# name= 'Indian Pines'
y13 = [72.33,80.16,90.794375,94.14,97.21]
y15 = [91.65,92.14,97.26,97.86,98.86]
y17 = [98.32,98.36,98.35,99.23,99.68]
# y19 = [40.02, 70.73, 79.24, 81.16, 93.14, 95.91, 94.86, 99.13]
# y21 = [48.89, 73.29, 80.01, 89.98, 97.89, 96.54, 98.36, 99.02]

# SA
# name='Salinas'
# y13 = [93.32, 99.12, 99.34, 99.66, 99.73, 99.92, 99.95, 99.95]
# y15 = [94.34, 98.97, 99.46, 99.85, 99.83, 99.85, 99.89, 99.90]
# y17 = [96.03, 99.51, 99.68, 99.77, 99.96, 99.97, 99.98, 99.92]
# y19 = [97.08, 99.26, 99.67, 99.91, 99.96, 99.97, 99.98, 99.98]
# y21 = [97.31, 99.41, 99.77, 99.91, 99.97, 99.98, 99.97, 99.95]

# PU
# name = 'Pavia University'
# y13 = [69.91, 89.52, 93.18, 95.28, 97.53, 98.58, 97.81, 99.29]
# y15 = [86.91, 92.02, 96.20, 98.08, 98.44, 98.98, 99.39, 99.38]
# y17 = [85.67, 94.27, 96.83, 98.23, 98.82, 99.39, 99.61, 99.85]
# y19 = [86.09, 95.87, 97.75, 98.20, 99.32, 99.55, 99.56, 99.75]
# y21 = [87.51, 95.20, 97.99, 97.94, 98.45, 99.54, 99.41, 99.66]

# plt.plot(x, y, 'ro-')
# plt.plot(x, y1, 'bo-')
# pl.xlim(-1, 11) # 限定横轴的范围
# pl.ylim(-1, 110) # 限定纵轴的范围
plt.plot(x, y13, marker='o', mec='r', mfc='w', label=u'Indian Pines')
plt.plot(x, y15, marker='*', mec='b', ms=5, label=u'Pavia University')
plt.plot(x, y17, marker='D', mec='y', ms=5, label=u'Salinas')
# plt.plot(x, y19, marker='X', mec='c', ms=5, label=u'19*19')
# plt.plot(x, y21, marker='s', mec='g', ms=5, label=u'21*21')

plt.legend()  # 让图例⽣效
plt.xticks(x, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel("Train sample ratio(%)")  # X轴标签
plt.ylabel("OA(%)")  # Y轴标签
# plt.title("{0} Classification Overall accuracies".format(name))  # 标题
plt.show()
