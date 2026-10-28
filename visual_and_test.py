import os
import numpy as np
from thop import profile, clever_format
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as tfs
from torchvision.utils import make_grid
import torchvision.utils as vutils

from models.model import Dehaze
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='its', help='its or nh')
parser.add_argument('--test_path', type=str, default='./test_images', help='Test imgs folder')
parser.add_argument('--output_path', type=str, default='./results', help='Output result folder')
parser.add_argument('--pretrained_model', type=str, default='./weights/ITS_train_37.54_0.9896.pth', help='Pretrained model weights')
parser.add_argument('--model_name', type=str, default='sgadn', help='Model name')
opt = parser.parse_args()


def tensorShow(tensors, titles=['haze']):
    fig = plt.figure()
    for tensor, tit, i in zip(tensors, titles, range(len(tensors))):
        img = make_grid(tensor)
        npimg = img.numpy()
        ax = fig.add_subplot(221 + i)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(tit)
    plt.show()


def save_images(model, pad=4):
    # plt.rcParams['figure.figsize'] = (12,9)
    output_dir = os.path.join(opt.output_path, opt.task)
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(opt.test_path, opt.task)
    for im in os.listdir(img_dir):
        print(f'\r {im}', end='', flush=True)
        haze = Image.open(os.path.join(img_dir, im))
        w = haze.size[1] - haze.size[1] % pad
        h = haze.size[0] - haze.size[0] % pad
        haze1 = tfs.Compose([
            tfs.ToTensor(),
            tfs.CenterCrop((w, h))
        ])(haze)[None, ::]
        '''
        haze_no = tfs.Compose([
            tfs.ToTensor(),
            tfs.CenterCrop((w, h))
        ])(haze)[None, ::]
        '''
        haze1 = haze1.to(device)
        with torch.no_grad():
            pred = model(haze1, use_eca=True)
            ts = torch.squeeze(pred.clamp(0, 1).cpu())
            # tensorShow([haze_no,pred.clamp(0,1).cpu()],['haze','pred'])
            vutils.save_image(ts, os.path.join(output_dir, f'{im.split('.')[0]}_{opt.model_name}.png'))


def analyze_model(model, input_tensor):
    """
    计算并打印模型的参数量和 FLOPs
    """
    model.eval()
    try:
        # 核心计算
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.3f")

        print(f"📥 Input Shape     : {tuple(input_tensor.shape)}")
        print(f"📊 Total Params   : {params_str}")
        print(f"⚡ Total FLOPs    : {flops_str}")
        print(f"🧮 MACs (Approx)  : {clever_format(flops / 2, '%.3f')[0]}")
        return params_str, flops_str

    except Exception as e:
        print(f"❌ 计算失败，请检查模型前向传播是否有报错: {e}")
        return None, None


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pk = torch.load(opt.pretrained_model)
    model = Dehaze().to(device)
    model.load_state_dict(pk)
    model.eval()

    save_images(model)

    INPUT_TENSOR = torch.randn(1, 3, 256, 256).to(device)
    analyze_model(model, INPUT_TENSOR)

