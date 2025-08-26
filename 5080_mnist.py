import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==================== 1. GPU极致优化配置 ====================
torch.backends.cudnn.benchmark = True  # 启用CuDNN自动优化器，根据输入尺寸自动选择最优卷积算法
torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32加速，在Ampere架构及以上的GPU上提供更快的矩阵运算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(f"Using device: {device} | GPU: {torch.cuda.get_device_name(0)}")

# ==================== 2. 数据预处理流水线 ====================
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像转换为Tensor，并自动归一化到[0,1]范围
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化：减均值0.1307，除标准差0.3081
])

# ==================== 3. 数据集加载与优化 ====================
# MNIST数据集：28x28灰度手写数字图像，10个类别(0-9)
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(
    train_data, 
    batch_size=1024,  # 大批次充分利用GPU并行计算能力
    shuffle=True,     # 每个epoch打乱数据顺序，防止模型记忆顺序
    num_workers=4,    # 4个进程并行加载数据，避免数据加载成为瓶颈
    pin_memory=True,  # 使用锁页内存，加速CPU到GPU的数据传输
    persistent_workers=True  # 保持worker进程存活，避免重复创建销毁的开销
)

test_loader = DataLoader(
    test_data, 
    batch_size=2048,  # 测试时使用更大批次（无需反向传播）
    num_workers=4,
    pin_memory=True
)

# ==================== 4. 神经网络模型定义 ====================
class Net(nn.Module):
    """
    卷积神经网络(CNN)用于手写数字识别
    结构：输入 -> [Conv2d -> GELU -> MaxPool] x2 -> Flatten -> Dropout -> Linear -> GELU -> Linear -> 输出
    """
    def __init__(self):
        super().__init__()
        # 第一卷积层：1输入通道(灰度图) -> 64输出通道，3x3卷积核，padding=1保持尺寸
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        # 第二卷积层：64输入通道 -> 128输出通道
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        # 全连接层1：将卷积特征展平后连接
        # 输入尺寸计算：经过两次2x2最大池化，28x28 -> 14x14 -> 7x7，128个通道
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        # 全连接层2（输出层）：512特征 -> 10类别输出
        self.fc2 = nn.Linear(512, 10)
        # Dropout层：20%的神经元随机失活，防止过拟合
        self.dropout = nn.Dropout(0.2)
        # GELU激活函数：高斯误差线性单元，比ReLU更平滑，性能更好
        self.gelu = nn.GELU()

    def forward(self, x):
        """
        前向传播过程：
        x: 输入张量 [batch_size, 1, 28, 28]
        返回: 输出logits [batch_size, 10]
        """
        # 第一卷积块
        x = self.gelu(self.conv1(x))  # [batch, 64, 28, 28]
        x = torch.max_pool2d(x, 2)    # [batch, 64, 14, 14] (2x2最大池化)
        
        # 第二卷积块  
        x = self.gelu(self.conv2(x))  # [batch, 128, 14, 14]
        x = torch.max_pool2d(x, 2)    # [batch, 128, 7, 7]
        
        # 展平特征图
        x = x.view(-1, 128 * 7 * 7)   # [batch, 128*7*7=6272]
        x = self.dropout(x)           # 训练时随机丢弃20%神经元
        
        # 全连接层
        x = self.gelu(self.fc1(x))    # [batch, 512]
        x = self.fc2(x)               # [batch, 10] (输出logits)
        
        return x

# 实例化模型并移动到GPU
model = Net().to(device)

# ==================== 5. 训练配置 ====================
# 混合精度训练：使用GradScaler管理FP16训练的梯度缩放，防止梯度下溢
scaler = torch.amp.GradScaler()

# 优化器：AdamW (Adam with Weight Decay)
# lr=0.001: 学习率，weight_decay=0.01: L2正则化强度，防止过拟合
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# ==================== 6. 训练和测试函数 ====================
def train(epoch):
    """训练一个epoch"""
    model.train()  # 设置为训练模式（启用Dropout等）
    for batch_idx, (data, target) in enumerate(train_loader):
        # 异步数据传输：non_blocking=True允许CPU和GPU并行工作
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        # 混合精度前向传播
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            output = model(data)  # 前向计算
            loss = nn.CrossEntropyLoss()(output, target)  # 计算交叉熵损失
        
        # 反向传播和优化
        scaler.scale(loss).backward()  # 缩放损失并反向传播
        scaler.step(optimizer)         # 更新参数
        scaler.update()                # 更新缩放因子
        optimizer.zero_grad(set_to_none=True)  # 高效清零梯度
        
        # 进度打印
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

def test():
    """测试模型性能"""
    model.eval()  # 设置为评估模式（禁用Dropout等）
    correct = 0
    # 禁用梯度计算 + 混合精度推理
    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)  # 获取预测类别
            correct += (pred == target).sum().item()  # 统计正确预测数
    
    # 计算并输出准确率
    acc = 100. * correct / len(test_loader.dataset)
    print(f"\nTest Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)\n")

# ==================== 7. 训练循环 ====================
for epoch in range(1, 11):  # 训练10个epoch
    train(epoch)
    if epoch % 2 == 0:  # 每2个epoch测试一次性能
        test()

# 保存训练好的模型参数
torch.save(model.state_dict(), "mnist_cnn_optimized.pt")