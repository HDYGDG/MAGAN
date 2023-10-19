import scipy.io as sio
import torchvision
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import spectral
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np

from MAE.auxil import AA_andEachClassAccuracy

np.seterr(divide='ignore',invalid='ignore')

from torch.autograd import Variable

from MAE_GAN import netD,netG,MaskedAutoencoderViT

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.chdir("../")


# now = time.strftime("%m%d%H%M", time.localtime())
# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def applyPCA2(X2, numComponents):
    newX = np.reshape(X, (-1, X.shape[0]))
    pca = PCA(n_components=145, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (145, X.shape[1], X.shape[2]))
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


# 用于测试样本的比例
test_ratio = 0.90
ratio = 10
# 每个像素周围提取 patch 的尺寸
patch_size = 21
# 使用 PCA 降维，得到主成分的数量
pca_components = 30
epoch = 100
dataname = 'IP'
os_name = 'Gan_base'
# IP  SA  PU  SAA
print("当前：{0}".format(dataname))
batch_size = 128
lr=0.00003



data_path = os.path.join(os.getcwd(), 'data')

X = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
y = labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
class_num = 16

# X = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
# y = labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
# class_num = 16
# (512, 217, 224)
#
# X = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
# y = labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
# class_num = 9
# (610, 340, 30)

# X = sio.loadmat(os.path.join(data_path, 'SalinasA_corrected.mat'))['salinasA_corrected']
# y = labels = sio.loadmat(os.path.join(data_path, 'SalinasA_gt.mat'))['salinasA_gt']
# class_num = 6
# (83, 86, 204)

x_pre = X
y_pre = y

print('Hyperspectral data shape: ', X.shape)
print('Label shape: ', y.shape)

print('\n... ... PCA tranformation ... ...')
X_pca = applyPCA(X, numComponents=pca_components)
print('Data shape after PCA: ', X_pca.shape)
print('\n... ... create data cubes ... ...')
X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)
print('Data cube X shape: ', X_pca.shape)
print('Data cube y shape: ', y.shape)
bands = X_pca.shape[-1];
numberofclass = len(np.unique(y))
print('\n... ... create train & test data ... ...')
Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, test_ratio)

print('Xtrain shape: ', Xtrain.shape)
print('Xtest  shape: ', Xtest.shape)

# 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components)
Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components)

Xtrain1 = Xtrain.transpose(0, 3, 1, 2)
Xtest1 = Xtest.transpose(0, 3, 1, 2)

print('Xtrain1 shape: ', Xtrain1.shape)
print('Xtest1  shape: ', Xtest1.shape)


class TrainDS1(torch.utils.data.Dataset):
    def __init__(self):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain1)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


# """ Testing dataset"""
class TestDS1(torch.utils.data.Dataset):
    def __init__(self):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest1)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


# 创建 trainloader 和 testloader
trainset1 = TrainDS1()
testset1 = TestDS1()

train_loader1 = torch.utils.data.DataLoader(drop_last=False,dataset=trainset1, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader1 = torch.utils.data.DataLoader(drop_last=False,dataset=testset1, batch_size=batch_size, shuffle=False, num_workers=0)

# 创建train_acc.csv和var_acc.csv文件，记录loss和accuracy
df = pd.DataFrame(columns=['step', 'train_loss'])  # 列名
df.to_csv("Gan_base/{0}/{0}_loss".format(dataname), index=False)  # 路径可以根据需要更改

nz = 256
ngf = 16
nc = 30
netG = netG(nz, ngf, nc, img_size=21, patch_size=3, in_chans=30,
            embed_dim=240, depth=2, num_heads=16,
            decoder_embed_dim=1024, decoder_depth=4, decoder_num_heads=8,
            mlp_ratio=4.).to(device)

ndf1 = 64
nc = 30
nb_label = 16
netD = netD(ndf=ndf1, nc=nc, nb_label=nb_label).to(device)

# d_optim = torch.optim.Adam(netG.parameters(), lr=0.0001)
# d_optim = torch.optim.Adam(netD.parameters(), lr=0.0001)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.02)
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.005)


s_criterion = nn.BCELoss()
c_criterion = nn.NLLLoss()

# input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
input = torch.FloatTensor(batch_size).to(device)
s_label = torch.FloatTensor(batch_size).to(device)
c_label = torch.FloatTensor(batch_size).to(device)
f_label = torch.FloatTensor(batch_size).to(device)

# c_label = Variable(c_label).to(device)
c_label = torch.tensor(labels, dtype=torch.long).to(device)

def kappa(testData, k):
    dataMat = np.mat(testData)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i] * 1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    Pe = float(ysum * xsum) / np.sum(dataMat) ** 2
    P0 = float(P0 / np.sum(dataMat) * 1.0)
    cohens_coefficient = float((P0 - Pe) / (1 - Pe))
    return cohens_coefficient


def gen_img_plot(model, epoch, test_input):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    fig = plt.figure(figsize=(4,4))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow((prediction[i]+1)/2)
        plt.axis('off')
    plt.show()

test_input = torch.randn(16,100).to(device)

D_loss=[]
G_loss=[]

nb_label=class_num

def show_image(image, title=''):
    # image is [H, W, 3]
    image = image.permute(1,2,0)
    image=applyPCA(image, numComponents=3)
    # print(image.shape)
    assert image.shape[2] == 3
    plt.imshow(np.clip((image) * 255, 0, 255)/255)
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def run_one_image(img1, img2, img3):
    x1 = torch.tensor(img1)
    x2 = torch.tensor(img2)
    x3 = torch.tensor(img3)
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.subplot(1, 3, 1)
    # print(x1[0].shape)
    show_image(x1[0], "fake1")

    plt.subplot(1, 3, 2)
    # print(x2[0].shape)
    show_image(x2[0], "fake2")

    plt.subplot(1, 3, 3)
    # print(x3[0].shape)
    show_image(x3[0], "real")
    plt.show()

def Test1(predict, labels):
    correct = 0
    pred = predict.data.max(1)[1]
    correct = pred.eq(labels.data).cpu().sum()
    return correct, len(labels.data)

decreas_lr='120,240,420,620,800'
decreasing_lr = list(map(int, decreas_lr.split(',')))
best_acc = 0
start = time.perf_counter()
count=0
print("开始训练")
for epoch in range(1, epoch + 1):
    netD.train()
    netG.train()
    right = 0
    estart = time.perf_counter()
    if epoch in decreasing_lr:
        optimizerD.param_groups[0]['lr'] *= 0.9
        optimizerG.param_groups[0]['lr'] *= 0.9
    for i, (inputs, labels) in enumerate(train_loader1):
        for j in range(10):    ## Update D 10 times for every G epoch

            netD.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = inputs.size(0)
            input.resize_(inputs.size()).copy_(inputs)
            # s_label.resize_(batch_size).fill_(labels)
            c_label.resize_(batch_size).copy_(labels)
            c_output = netD(input)

            # s_errD_real = s_criterion(s_output, s_label)
            c_errD_real = c_criterion(c_output, c_label)
            errD_real = c_errD_real
            errD_real.backward()
            D_x = c_output.data.mean()

            correct, length = Test1(c_output, c_label)
            # print('real train finished!')

            # label = np.random.randint(0, nb_label, batch_size)
            label = np.full(batch_size, nb_label)

            f_label.resize_(batch_size).copy_(torch.from_numpy(label))

            fake1, fake2, mae_loss = netG(input,0.4)
            # s_label.fill_(fake_label)
            c_output1 = netD(fake1.detach())
            c_output2 = netD(fake2.detach())
            # s_errD_fake = s_criterion(s_output, s_label)
            c_errD_fake1 = c_criterion(c_output1, f_label.long())
            c_errD_fake2 = c_criterion(c_output2, f_label.long())
            errD_fake = c_errD_fake1+c_errD_fake2
            errD_fake.backward()
            D_G_z1 = c_output1.data.mean()
            D_G_z2 = c_output2.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()
            # print('fake train finished!')
            ###############
            #  Updata G
            ##############

        netG.zero_grad()
        # s_label.data.fill_(real_label)  # fake labels are real for generator cost
        c_output1 = netD(fake1)
        c_output2 = netD(fake2)
        # s_errG = s_criterion(s_output, s_label)
        c_errG1 = c_criterion(c_output1, c_label)
        c_errG2 = c_criterion(c_output2, c_label)
        errG = (c_errG1+c_errG2)/2
        errG.backward()
        D_G_z3 = c_output1.data.mean()
        D_G_z4 = c_output2.data.mean()
        optimizerG.step()
        right += correct
        if epoch % 10 == 0:
            fake_img1 = fake1.data.cpu().numpy()
            fake_img2 = fake2.data.cpu().numpy()
            real_img = input.data.cpu().numpy()
            run_one_image(fake_img1,fake_img2, real_img)
        # print('begin spout!')

    if epoch % 5 == 0:
        print('[%d/%d][%d/%d]   D(x): %.4f D(G(z)): %.4f / %.4f=%.4f,  Accuracy: %.4f / %.4f = %.4f'
              % (epoch, epoch, i, len(train_loader1),
                 D_x, D_G_z1+D_G_z2, D_G_z3+D_G_z4, D_G_z1+D_G_z2 / D_G_z3+D_G_z4, right, len(train_loader1.dataset), 100. * right / len(train_loader1.dataset)))

    # torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    # torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    if epoch % 5 == 0:
        netD.eval()
        netG.eval()
        test_loss = 0
        right = 0
        all_Label = []
        all_target = []
        for i, (data, target) in enumerate(test_loader1):
            indx_target = target.clone()
            if torch.cuda.is_available():
                data, target = data.to(device), target.to(device)
            with torch.no_grad():
                data, target = Variable(data), Variable(target)

            # fake=netG(data)
            # output = netD(data)
            # vutils.save_image(data,'./img/%s/real_samples_i_%03d.png' % (opt.outf,epoch))
            # vutils.save_image(fake,'./img/%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch))
            output = netD(data)

            outputs = np.argmax(output.detach().cpu().numpy(), axis=1)
            if count == 0:
                y_pred_test = outputs
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))

            test_loss += c_criterion(output, target).item()
            # output1=output[:,0:16]
            pred = output.max(1)[1]  # get the index of the max log-probabilityo
            all_Label.extend(pred.cpu())
            all_target.extend(target.cpu())
            right += pred.cpu().eq(indx_target).sum()

        test_loss = test_loss / len(test_loader1)  # average over number of mini-batch
        acc = float(100. * float(right)) / float(len(test_loader1.dataset))
        print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, right, len(test_loader1.dataset), acc))
        if acc > best_acc:
            best_acc = acc
        # C = confusion_matrix(target.data.cpu().numpy(), pred.cpu().numpy())
        C = confusion_matrix(all_target, all_Label)
        C = C[:class_num, :class_num]
        np.save('c.npy', C)
        k = kappa(C, np.shape(C)[0])
        AA_ACC = (np.diag(C) / np.sum(C, 1))
        AA = np.mean(AA_ACC,0)
        print('OA= %.5f AA= %.5f k= %.5f' % (acc, AA*100, k*100))

#
end = time.perf_counter()
# 读取csv中指定列的数据
data = pd.read_csv("Gan_base/{0}/{0}_loss".format(dataname))


count = 0
# 模型测试
# net.eval()  # 注意启用测试模 式
for inputs, _ in test_loader1:
    inputs = inputs.to(device)
    outputs = netD(inputs)
    outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
    if count == 0:
        y_pred_test = outputs
        count = 1
    else:
        y_pred_test = np.concatenate((y_pred_test, outputs))
# 生成分类报告
classification = classification_report(ytest, y_pred_test, digits=4)
print(classification)

from operator import truediv


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc
#
#
def reports(test_loader1, y_test, name):
    count = 0
    # 模型测试
    for inputs, _ in test_loader1:
        inputs = inputs.to(device)
        outputs = netD(inputs)

        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred = outputs
            count = 1
        else:
            y_pred = np.concatenate((y_pred, outputs))

    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth',
                        'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
    elif name == 'PU':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
    elif name == 'SAA':
        target_names = ['Brocoli_green_weeds_1', 'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk',
                        'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_8wk']

    classification = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)
    print("acc:")
    return classification, confusion, oa * 100, each_acc * 100, aa * 100, kappa * 100
#
#
classification, confusion, oa, each_acc, aa, kappa = reports(test_loader1, ytest, dataname)
#
classification = str(classification)
confusion = str(confusion)
file_name = "Gan_base/{0}/{0}_Record".format(dataname)
#
torch.save(netD, "Gan_base/model/{0}/{0}_model.pth".format(dataname))
with open(file_name, 'w') as x_file:
    x_file.write('\n')
    x_file.write('{} Kappa accuracy (%)'.format(kappa))
    x_file.write('\n')
    x_file.write('{} Overall accuracy (%)'.format(oa))
    x_file.write('\n')
    x_file.write('{} Average accuracy (%)'.format(aa))
    x_file.write('\n')
    x_file.write('{} 运行时间 (s)'.format(end - start))
    x_file.write('\n')
    x_file.write('{}'.format(classification))
    x_file.write('\n')
    x_file.write('{}'.format(confusion))

X = x_pre
y = y_pre
height = y.shape[0]
width = y.shape[1]

X = applyPCA(X, numComponents=pca_components)
X = padWithZeros(X, patch_size // 2)

# 逐像素预测类别
print('逐像素预测类别')
outputs = np.zeros((height, width))

for i in range(height):
    for j in range(width):
        if int(y[i, j]) == 0:
            continue
        else:
            image_patch = X[i:i + patch_size, j:j + patch_size, :]
            image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2])
            X_test_image = torch.FloatTensor(image_patch.transpose(0, 3, 1, 2)).to(device)
            prediction = netD(X_test_image)
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
                     [0, 254, 254],
                     ])

ground_truth = spectral.imshow(classes=outputs.astype(int), figsize=(100, 100), colors=ip_color)
spectral.save_rgb('Gan_base/{0}/{0}_pre_complete.jpg'.format(dataname), outputs.astype(int), colors=ip_color)
print("当前：{0}_{0}".format(dataname))
