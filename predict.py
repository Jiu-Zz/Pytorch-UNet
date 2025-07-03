import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def predict_img(net,
                full_img,
                device,
                transform):
    net.eval()
    img = transform(full_img).unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')

    output = (output * 0.5 + 0.5).clamp(0, 1)

    output_np = output.squeeze(0).permute(1, 2, 0).numpy()
    output_np = (output_np * 255).astype(np.uint8)
    deblurred_img = Image.fromarray(output_np)

    return deblurred_img


def get_args():
    parser = argparse.ArgumentParser(description='Predict sharp images from blur images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='模型的存储路径 (default: MODEL.pth)')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='输入的文件名', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='输出的文件名',)
    parser.add_argument('--no-save', '-n', action='store_true', help='不保存输出图像')
    parser.add_argument('--bilinear', action='store_true', default=False, help='使用双线性插值')
    
    return parser.parse_args()

def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'
    return args.output or list(map(_generate_name, args.input))

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, out_channels=3, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        deblurred_img = predict_img(net=net,
                           full_img=img,
                           device=device,
                           transform=transform)

        if not args.no_save:
            out_filename = out_files[i]
            deblurred_img.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')