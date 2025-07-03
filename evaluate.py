import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from deblur_dataset_with_split import DeblurDataset
from unet import UNet

dir_img = Path('./data')
dir_checkpoint = Path('./checkpoints/')

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小为U-Net常用的输入尺寸
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1, 1]范围
])

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    n_val = len(dataloader)
    total_psnr = 0.0
    total_ssim = 0.0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        total_val_images = n_val * dataloader.batch_size
        with tqdm(total=total_val_images, desc='Validation', unit='img') as pbar:
            for batch in dataloader:
                blurred_image, sharp_image = batch[0], batch[1]

                # 转移到设备（保持与训练一致的内存格式）
                blurred_image = blurred_image.to(device=device, dtype=torch.float32)
                sharp_image = sharp_image.to(device=device, dtype=torch.float32)

                # 模型推理
                deblurred_image = net(blurred_image)

                deblurred_image = (deblurred_image * 0.5 + 0.5).clamp(0.0, 1.0)
                sharp_image = (sharp_image * 0.5 + 0.5).clamp(0.0, 1.0)

                # 转换为numpy并调整通道顺序 (B, C, H, W) → (B, H, W, C)
                deblurred_np = deblurred_image.cpu().numpy().transpose(0, 2, 3, 1)
                sharp_np = sharp_image.cpu().numpy().transpose(0, 2, 3, 1)

                # 计算每个样本的PSNR和SSIM
                for i in range(deblurred_np.shape[0]):
                    psnr = peak_signal_noise_ratio(
                        sharp_np[i], deblurred_np[i], data_range=1.0
                    )
                    ssim = structural_similarity(
                        sharp_np[i],
                        deblurred_np[i],
                        data_range=1.0,
                        multichannel=True,
                        channel_axis=2,
                        win_size=7
                    )
                    total_psnr += psnr
                    total_ssim += ssim

                pbar.update(blurred_image.shape[0])

    avg_psnr = total_psnr / total_val_images
    avg_ssim = total_ssim / total_val_images

    net.train()
    return {'psnr': avg_psnr, 'ssim': avg_ssim}

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate the UNet on blurred images and sharp images')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='批次大小')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='模型的存储路径 (default: MODEL.pth)')
    parser.add_argument('--amp', action='store_true', default=False, help='使用自动混合精度训练')
    parser.add_argument('--bilinear', action='store_true', default=False, help='使用双线性插值上采样')

    return parser.parse_args()

if __name__ == '__main__':
    # 0. 参数解析
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 模型
    model = UNet(n_channels=3, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    # 2.1 创建数据集
    try:
        val_set = DeblurDataset(
            root_dir=dir_img,
            transform=transform,
            mode='test',
            augment=True
        )
    except (AssertionError, RuntimeError, IndexError):
        val_set = None
        print('Dataset not found or corrupted. ')

    # 2.2 创建数据加载器
    n_val = len(val_set)
    loader_args = dict(batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 3. 加载模型
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device=device)

    # 4. 评估模型
    result = evaluate(net=model,
             dataloader=val_loader,
             device=device,
             amp=args.amp)
    print(result)