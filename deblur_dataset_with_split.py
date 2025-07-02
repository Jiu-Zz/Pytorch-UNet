import os
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import random
from sklearn.model_selection import train_test_split

class DeblurDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train', augment=True):
        """
        初始化图像去模糊数据集
        
        参数:
            root_dir: 数据集根目录，应包含train和test子目录
            transform: 应用于图像的变换
            mode: 数据集模式，'train'或'val'或'test'
            augment: 是否应用数据增强
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.augment = augment and mode == 'train'  # 只在训练模式下应用增强
        
        # 根据模式设置数据集路径
        if mode in ['train', 'val']:
            self.blurred_dir = os.path.join(root_dir, 'train', 'blurred')
            self.sharp_dir = os.path.join(root_dir, 'train', 'sharp')
        else:  # test模式
            self.blurred_dir = os.path.join(root_dir, 'test', 'blurred')
            self.sharp_dir = os.path.join(root_dir, 'test', 'sharp')
        
        # 获取所有图像文件名
        self.image_files = [f for f in os.listdir(self.blurred_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # 确保模糊图像和清晰图像数量相同
        assert len(self.image_files) == len(os.listdir(self.sharp_dir)), \
            "模糊图像和清晰图像数量不匹配"
    
    def __len__(self):
        """返回数据集的样本数量"""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        获取指定索引的样本对（模糊图像和清晰图像）
        
        参数:
            idx: 样本索引
        
        返回:
            blurred_image: 模糊图像
            sharp_image: 对应的清晰图像
        """
        # 获取图像文件名
        img_name = self.image_files[idx]
        
        # 加载模糊图像和清晰图像
        blurred_path = os.path.join(self.blurred_dir, img_name)
        sharp_path = os.path.join(self.sharp_dir, img_name)
        
        blurred_image = Image.open(blurred_path).convert('RGB')
        sharp_image = Image.open(sharp_path).convert('RGB')
        
        # 应用数据增强（仅在训练模式下）
        if self.augment:
            # 随机水平翻转
            if random.random() > 0.5:
                blurred_image = blurred_image.transpose(Image.FLIP_LEFT_RIGHT)
                sharp_image = sharp_image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # 随机垂直翻转
            if random.random() > 0.5:
                blurred_image = blurred_image.transpose(Image.FLIP_TOP_BOTTOM)
                sharp_image = sharp_image.transpose(Image.FLIP_TOP_BOTTOM)
            
            # 随机旋转90度
            if random.random() > 0.75:
                angle = random.choice([90, 180, 270])
                blurred_image = blurred_image.rotate(angle)
                sharp_image = sharp_image.rotate(angle)
        
        # 应用变换
        if self.transform:
            blurred_image = self.transform(blurred_image)
            sharp_image = self.transform(sharp_image)
            
        return blurred_image, sharp_image

# 辅助函数：创建训练集和验证集
def create_train_val_datasets(root_dir, transform, val_size=0.2, random_state=42):
    """
    创建训练集和验证集
    
    参数:
        root_dir: 数据集根目录
        transform: 图像变换
        val_size: 验证集比例
        random_state: 随机种子，确保结果可复现
    
    返回:
        train_dataset: 训练集
        val_dataset: 验证集
    """
    # 创建完整的训练集
    full_train_dataset = DeblurDataset(
        root_dir=root_dir,
        transform=transform,
        mode='train',
        augment=True
    )
    
    # 获取所有样本索引
    indices = list(range(len(full_train_dataset)))
    
    # 随机划分训练集和验证集
    train_indices, val_indices = train_test_split(
        indices, 
        test_size=val_size, 
        random_state=random_state
    )
    
    # 创建Subset对象
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)
    
    # 为验证集设置适当的augment参数
    # 注意：Subset对象没有直接的属性访问权限，我们需要修改原始数据集
    full_train_dataset.augment = False  # 禁用原始数据集的增强
    val_dataset.dataset.augment = False  # 确保验证集不使用增强
    
    # 重新启用训练集的增强
    train_dataset.dataset.augment = True
    
    return train_dataset, val_dataset

# 数据预处理变换
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小为U-Net常用的输入尺寸
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1, 1]范围
])

# 创建训练集和验证集
train_dataset, val_dataset = create_train_val_datasets(
    root_dir='data',
    transform=transform,
    val_size=0.2,  # 20%的数据用于验证
    random_state=42  # 固定随机种子，确保结果可复现
)

# 创建测试集
test_dataset = DeblurDataset(
    root_dir='data',
    transform=transform,
    mode='test',
    augment=False
)

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=4,  # 根据你的GPU内存调整批量大小
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# 测试数据加载器
if __name__ == "__main__":
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    for blurred_imgs, sharp_imgs in train_loader:
        print(f"训练集批次 - Blurred images shape: {blurred_imgs.shape}")
        print(f"训练集批次 - Sharp images shape: {sharp_imgs.shape}")
        break
    
    for blurred_imgs, sharp_imgs in val_loader:
        print(f"验证集批次 - Blurred images shape: {blurred_imgs.shape}")
        print(f"验证集批次 - Sharp images shape: {sharp_imgs.shape}")
        break
    
    for blurred_imgs, sharp_imgs in test_loader:
        print(f"测试集批次 - Blurred images shape: {blurred_imgs.shape}")
        print(f"测试集批次 - Sharp images shape: {sharp_imgs.shape}")
        break
