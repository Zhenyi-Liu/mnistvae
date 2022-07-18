import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# VAE模型
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=2):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)

    # 编码，学习高斯分布均值与方差，卷积生成2*20个数，分别是mu及std（计算后可以得到），然后
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    # 将高斯分布均值与方差参数重表示，生成隐变量z  若x~N(mu, var*var)分布,则(x-mu)/var=z~N(0, 1)分布
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2) #log_var是用来计算方差的
        eps = torch.randn_like(std)
        return mu + eps * std

    # 解码隐变量z
    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))

    # 计算重构值和隐变量z的分布参数
    def forward(self, x):
        mu, log_var = self.encode(x)  # 从原始样本x中学习隐变量z的分布，即学习服从高斯分布均值与方差
        z = self.reparameterize(mu, log_var)  # 将高斯分布均值与方差参数重表示，生成隐变量z
        x_reconst = self.decode(z)  # 解码隐变量z，生成重构x’
        return x_reconst, mu, log_var,z  # 返回重构值和隐变量的分布参数

    # 横坐标直接用图像表达的数字。fc2(h)=mu, fc3(h)=log_var为了获取高斯曲线用的。
    # 纵坐标是z。
