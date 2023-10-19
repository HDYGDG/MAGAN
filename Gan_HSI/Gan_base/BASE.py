import torch
import torch.nn as nn

from Gan_base.Gan_Tran import SSFTTnet

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Ssftt(nn.Module):

    def __init__(self):
        super(Ssftt, self).__init__()
        self.net = SSFTTnet().to(device)

    def forward(self, x):
        y=self.net(x)

        return y


if __name__ == '__main__':
    # 随机输入，测试网络结构是否通
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    x = torch.randn(128,1, 30, 15, 15).to(device)


    net = Ssftt().to(device)

    y = net(x)
    # print(y.shape)
