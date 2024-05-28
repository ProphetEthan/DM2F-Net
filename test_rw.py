import os
import torch
from torch import nn
from torchvision import transforms

from tools.config import TEST_SOTS_ROOT, OHAZE_ROOT, TEST_Haze_ROOT
from tools.utils import AvgMeter, check_mkdir, sliding_forward
from model import DM2FNet, DM2FNet_o
from datasets import SotsDataset, OHazeDataset, HazeDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import deltaE_ciede2000, rgb2lab
from PIL import Image


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

torch.manual_seed(2018)
torch.cuda.set_device(0)

to_pil = transforms.ToPILImage()

def main():
    with torch.no_grad():
        net = DM2FNet_o().cuda()
        # snapshot = './ckpt/RESIDE_ITS/iter_20000_loss_0.03197_lr_0.000268.pth'
        snapshot = './ckpt_o/RESIDE_ITS/iter_40000_loss_0.01487_lr_0.000000.pth'
        print('load snapshot \'%s\' for testing',snapshot)
        net.load_state_dict(torch.load(snapshot, map_location="cuda:0"))

        net.eval()
        
        for idx in range(5):

            haze = Image.open(f'./data/my_pic/pic_{idx+1}.jpg')

            transform = transforms.Compose([
               
                transforms.ToTensor()
            ])
            
            haze = transform(haze)
            check_mkdir('./output_o')

            haze = haze.cuda()

           
            res = net(haze.view(1, 3, haze.shape[1], -1)).detach()

            res = res.cpu()
            to_pil(res.view(3, res.shape[2], -1)).save(f'./output_o/pic_{idx+1}.jpg')

if __name__ == '__main__':
    main()
