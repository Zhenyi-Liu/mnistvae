import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms,datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from model import VAE
import matplotlib.pyplot as plt
import numpy as np

# dataset = torchvision.datasets.MNIST(root='./data',
#                                      train=True,
#                                      transform=transforms.ToTensor(),
#                                      download=True)
#
# # 数据加载，按照batch_size大小加载，并随机打乱
# data_loader = torch.utils.data.DataLoader(dataset=dataset,
#                                           batch_size=batch_size,
#                                           shuffle=True)
# 读取数据,数据为自我设定非常灵活变更数据库就可以进行更多测试
# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1)
])
root = 'data'
train_dataset = datasets.ImageFolder(root + '/train',transform)
test_dataset = datasets.ImageFolder(root + '/test',transform)
# 导入数据,分批打乱有利于增加训练复杂度
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=80, shuffle=True,drop_last=True,num_workers=8, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=80, shuffle=False,drop_last=True,num_workers=8, pin_memory=True)

# 配置GPU或CPU设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建目录保存生成的图片
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# 超参数设置
image_size = 784  # 图片大小
h_dim = 400
z_dim = 2
num_epochs = 3000  # 15个循环
batch_size = 1  # 一批的数量
learning_rate = 1e-4  # 学习率

# 构造VAE实例对象
model = VAE().to(device)

# 选择优化器，并传入VAE模型参数和学习率
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
def train():

    model.train()
    # 开始训练一共15个循环
    for epoch in range(num_epochs):
        # y_np = []
        # z_np = []
        for i, (x, y) in enumerate(train_loader):
            # print("x::",x.size())
            # 前向传播
            x = x.to(device).view(-1,
                                  image_size)  # 将batch_size*1*28*28 ---->batch_size*image_size  其中，image_size=1*28*28=784
            # print("x------::",x.size())
            x_reconst, mu, log_var,z= model(x)  # 将batch_size*748的x输入模型进行前向传播计算,重构值和服从高斯分布的隐变量z的分布参数（均值和方差）
            # print("x_reconst:",x_reconst.size())
            # print("mu:",mu.size())
            # print("log_var:",log_var.size())
            # print("z:",z.size())
            # 记录latentspace输出情况
            y_cpu = y.cpu().detach().numpy()
            z_cpu = z.cpu().detach().numpy()
            with open("./latenttrain_output.txt", "a", encoding="utf-8") as f:
                f.write(str(y_cpu))
                f.write(":")
                f.write(str(z_cpu))
                f.write("\n")
            # 直接将每个y对应的是数字，z对应的是一个2维空间，代表latent_x轴latent_y轴，使用颜色标签来代表数字y。
            # 知道隐空间和输入数据的对应关系后就可以指定隐空间的值来确定输出结果的好坏。每一轮画一张图。
            # y_np.extend(y_cpu)
            # z_np.extend(z_cpu)
            # 计算重构损失和KL散度
            # 重构损失
            reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
            # KL散度
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            # 反向传播与优化
            # 计算误差(重构误差和KL散度值)
            loss = reconst_loss + kl_div
            # 清空上一步的残余更新参数值
            optimizer.zero_grad()
            # 误差反向传播, 计算参数更新值
            loss.backward()
            # 将参数更新值施加到VAE model的parameters上
            optimizer.step()
            # 每迭代一定步骤，打印结果值
            if (i + 1) % 10 == 0:
                print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, len(train_loader), reconst_loss.item(), kl_div.item()))

        with torch.no_grad():
            # 保存采样值
            # 生成随机数 z
            z = torch.randn(batch_size, z_dim).to(device)  # z的大小为batch_size * z_dim = 128*20
            # 对随机数 z 进行解码decode输出
            out = model.decode(z).view(-1, 1, 28, 28)
            # 保存结果值
            save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch + 1)))

            # 保存重构值
            # 将batch_size*748的x输入模型进行前向传播计算，获取重构值out
            out, _, _,_ = model(x)
            # 将输入与输出拼接在一起输出保存  batch_size*1*28*（28+28）=batch_size*1*28*56
            x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
            save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch + 1)))
    # plotdistribution(y_np, z_np)
        if epoch%100 ==0:
            torch.save(model.state_dict(), "./my_model_epoch{}.pth".format(epoch))  # 只保存模型的参数

    torch.save(model.state_dict(), "./my_model.pth")  # 只保存模型的参数

def evaluation():
    model.eval()
    model.load_state_dict(torch.load("./my_model_epoch100.pth"))
    y_np = []
    z_np = []
    for i, (x, y) in enumerate(test_loader):
        # 前向传播
        x = x.to(device).view(-1,
                              image_size)  # 将batch_size*1*28*28 ---->batch_size*image_size  其中，image_size=1*28*28=784
        x_reconst, mu, log_var,z = model(x)  # 将batch_size*748的x输入模型进行前向传播计算,重构值和服从高斯分布的隐变量z的分布参数（均值和方差）

        # 记录latentspace输出情况
        y_cpu = y.cpu().detach().numpy()
        z_cpu = z.cpu().detach().numpy()
        y_np.extend(y_cpu)
        z_np.extend(z_cpu)
        with open("./latent_output.txt","a",encoding="utf-8")as f:
            f.write(str(y_cpu))
            f.write(":")
            f.write(str(z_cpu))
            f.write("\n")

        # 计算重构损失和KL散度
        # 重构损失
        reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
        # KL散度
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # 反向传播与优化
        # 每迭代一定步骤，打印结果值
        if (i + 1) % 10 == 0:
            print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                  .format(0 + 1, 1, i + 1, len(test_loader), reconst_loss.item(), kl_div.item()))
    plotdistribution(y_np, z_np)

def plotdistribution(y_np,z_np=None):
    # 建立画布
    plt.figure(figsize=(9, 9))
    # 设置坐标轴范围
    # plt.xlim((-4, 12))
    # plt.ylim((-12, 12))
    # plt.xlabel('xxxxxxxxxxx')
    # plt.ylabel('yyyyyyyyyyy')
    # # 设置坐标轴刻度
    # my_x_ticks = np.arange(-4, 12, 0.5)
    # my_y_ticks = np.arange(-12, 12, 0.5)
    # plt.xticks(my_x_ticks)
    # plt.yticks(my_y_ticks)

    for index,label in enumerate(y_np):
        if label==0:
            x=z_np[index][0]
            y=z_np[index][1]
            plt.scatter(x,y,s=5, c='r')
        elif label==1:
            x=z_np[index][0]
            y=z_np[index][1]
            plt.scatter(x,y,s=5,  c='g')
        elif label==2:
            x=z_np[index][0]
            y=z_np[index][1]
            plt.scatter(x,y,s=5,  c='b')
        elif label==3:
            x=z_np[index][0]
            y=z_np[index][1]
            plt.scatter(x,y,s=5,  c='y')
        elif label==4:
            x=z_np[index][0]
            y=z_np[index][1]
            plt.scatter(x,y, s=5, c='k')
        elif label==5:
            x=z_np[index][0]
            y=z_np[index][1]
            plt.scatter(x,y,s=5,  c='m')
        elif label==6:
            x=z_np[index][0]
            y=z_np[index][1]
            plt.scatter(x,y,s=5,  c='c')
        elif label==7:
            x=z_np[index][0]
            y=z_np[index][1]
            plt.scatter(x,y,s=5,  c='pink')
        elif label==8:
            x=z_np[index][0]
            y=z_np[index][1]
            plt.scatter(x,y,s=5,  c='grey')
        elif label==9:
            x=z_np[index][0]
            y=z_np[index][1]
            plt.scatter(x,y,s=5,  c='blueviolet')
    plt.show()

if __name__ == '__main__':

    # train()
    evaluation()
    # 隐空间可视化要求输入为数字输出为数字的隐空间输出。但是输出是20维，所以怎么图像展示？
    # 1）变更输出为1维