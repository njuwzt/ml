import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision
import numpy
from torch.autograd import Variable
from torchviz import make_dot

class normalCNN(nn.Module):
    #网络结构：卷积层，激活函数，池化层，全连接层
    def __init__(self):
        super(normalCNN,self).__init__()
        self.conv1 = nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(stride=2,kernel_size=2)
        self.fc1 = nn.Linear(14*14*128,1024)
        self.fc2 = nn.Linear(1024, 10)
        self.drop = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1,14*14*128)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    Transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.1307,],std=[0.3081,])])
    #下载训练集，测试集
    trainSet = datasets.MNIST(root="./data",transform=Transform,train=True,download=True)
    testSet = datasets.MNIST(root="./data",transform=Transform,train=False)
    #装载训练集，测试集，一个batch数据集64张图
    trainLoader = torch.utils.data.DataLoader(dataset=trainSet,batch_size = 64,shuffle = True,num_workers=0) 
    testLoader = torch.utils.data.DataLoader(dataset=testSet,batch_size = 64,shuffle = True,num_workers=0)
    model = normalCNN()
    model.to(device)
    #g=make_dot(model(torch.rand(64,1,28,28)),params=dict(model.named_parameters()))
    #g.view()
    #损失函数使用交叉熵
    cost = nn.CrossEntropyLoss()
    #优化计算方式选择：
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,betas=(0.9,0.99))
    epochSize = 20
    runLoss_list=[]
    testLoss_list=[]
    accuracy_list=[]
    for epoch in range(epochSize):
        runLoss=0.0
        trainCorrect=0
        print("Epoch {}/{}".format(epoch,epochSize-1))
        print("----------")
        #training:
        for trainData in trainLoader:
            inputs,labels = trainData
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)
            _,pred = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss = cost(outputs, labels)
            loss.backward()
            optimizer.step()
            runLoss += loss.item()
            trainCorrect += torch.sum(pred == labels.data)
        #testing:
        testCorrect=0
        testLoss=0.0
        for testData in testLoader:
            inputs,labels = testData
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)
            loss = cost(outputs, labels)
            testLoss += loss.item()
            _,pred = torch.max(outputs.data, 1)
            testCorrect += torch.sum(pred == labels.data)
        trainDataLen=len(trainSet)
        testDataLen=len(testSet)
        runLoss_list.append(runLoss/trainDataLen)
        testLoss_list.append(testLoss/testDataLen)
        accuracy_list.append(testCorrect*100/testDataLen)
        print("Train Accuracy is {:.4f}%,Test Accuracy is {:.4f}%,trainLoss is {:.4f},testLoss is {:.4f}".format(trainCorrect*100/trainDataLen,testCorrect*100/testDataLen,runLoss/trainDataLen,testLoss/testDataLen))
    plt.figure(1)
    x1=range(0,epochSize)
    x2=range(0,epochSize)
    x3=range(0,epochSize)
    y1=accuracy_list
    y2=runLoss_list
    y3=testLoss_list
    plt.plot(x1,y1,'o-')
    plt.title('Test accuracy in epoches')
    plt.ylabel('Test accuracy')
    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(x2, y2, '.-')
    plt.title('Train and Test Loss in epoches')
    plt.ylabel('Train loss')
    plt.subplot(2, 1, 2)
    plt.plot(x3, y3, '.-')
    plt.xlabel('epoches')
    plt.ylabel('Test loss')
    plt.show()
