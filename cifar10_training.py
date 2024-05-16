import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# 定义数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)

# 定义ResNet模型
resnet = models.resnet18(pretrained=False)
# print(resnet)# 有必要，原版是输出1000
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 10)  # 将全连接层改为10个输出类别
#print(resnet)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

# 训练模型
# 将模型移动到可用的设备上（优先使用GPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet.to(device)

# 设定训练的epoch数量，这里固定为10
for epoch in range(10):
    running_loss = 0.0
    # 遍历训练数据集
    for i, data in enumerate(trainloader, 0):
        # 将输入数据和标签移动到与模型相同的设备上
        inputs, labels = data[0].to(device), data[1].to(device)

        # 清除之前的梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = resnet(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播，计算梯度
        loss.backward()
        # 使用优化器更新模型参数
        optimizer.step()

        # 累加当前批次的损失
        running_loss += loss.item()
        # 每处理100个批次，输出一次当前平均损失
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
print('Finished Training')

# 该代码段用于测试预训练的resnet模型在给定测试数据集上的准确率

# 初始化正确预测数和总预测数
correct = 0
total = 0

# 在不计算梯度的环境下遍历测试数据集
with torch.no_grad():
    for data in testloader:
        # 将图片数据和标签转移到指定设备上
        images, labels = data[0].to(device), data[1].to(device)
        # 通过resnet模型预测图片类别
        outputs = resnet(images)
        # 获取预测结果中概率最大的类别索引
        _, predicted = torch.max(outputs.data, 1)
        # 更新总预测数
        total += labels.size(0)
        # 对比预测结果和真实标签，累计正确预测数
        correct += (predicted == labels).sum().item()

# 计算并打印整体准确率
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

