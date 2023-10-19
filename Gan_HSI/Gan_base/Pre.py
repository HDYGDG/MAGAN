import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import spectral
import torch

import os
import numpy as np

import time

from Gan_base.BASE import Ssftt


def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
        print(windowSize)
    return patchesData, patchesLabels

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    return X_train, X_test, y_train, y_test

data_path = '../data'
# X = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
# y = labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
# class_num = 16

# X = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
# y = labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
# class_num = 16
# (512, 217, 224)

X = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
y = labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
class_num = 9
# (610, 340, 30)

# X = sio.loadmat(os.path.join(data_path, 'SalinasA_corrected.mat'))['salinasA_corrected']
# y = labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
# class_num = 6
# (83, 86, 204)

dataname = 'PU'
# IP  SA  PU  SAA

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# 用于测试样本的比例
test_ratio = 0.99
ratio=1
# 每个像素周围提取 patch 的尺寸
patch_size = 15
# 使用 PCA 降维，得到主成分的数量
pca_components = 30
dirname = "../Gan_base"
net = torch.load('{0}/{1}/{1}_model.pth'.format(dirname, dataname)).to(device)

height = y.shape[0]
width = y.shape[1]

X = applyPCA(X, numComponents=pca_components)
X = padWithZeros(X, patch_size // 2)

# 逐像素预测类别
print("开始预测")
outputs = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        # if int(y[i, j]) == 0:
        #     continue
        # else:
            image_patch = X[i:i + patch_size, j:j + patch_size, :]
            image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],1)
            X_test_image = torch.FloatTensor(image_patch.transpose(0,4, 3, 1, 2)).to(device)
            prediction = net(X_test_image)
            # print(prediction.shape)
            prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
            outputs[i][j] = prediction + 1
    if i % 5 == 0:
        print('... ... row ', i, ' handling ... ...')

ip_color = np.array([[255, 255, 255],
                     [84, 171, 171],
                     [192, 20, 235],
                     [205, 231, 24],
                     [153, 102, 125],
                     [123, 123, 123],
                     [183, 40, 99],
                     [0, 39, 245],
                     [0, 176, 240],
                     [255, 255, 0],
                     [237, 125, 49],
                     [0, 32, 96],
                     [131, 60, 11],
                     [70, 114, 196],
                     [55, 86, 35],
                     [255, 0, 0],
                     [0, 254, 254]
                     ])

ground_truth = spectral.imshow(classes=outputs.astype(int), figsize=(100, 100), colors=ip_color)
now = time.strftime("%m%d%H%M", time.localtime())
spectral.save_rgb('{0}/{1}_pre_complete.jpg'.format(dirname, dataname), outputs.astype(int), colors=ip_color)
print("开始预测")