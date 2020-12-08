import os
import os.path

import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
import pdb
import math
import imageio
import numpy as np


def read_img():
    # file='/home/shctrl/yny_intern/image_conmpression/CVQN-master/image/HDR/1.hdr'
    file='image/HDR_MATLAB_3/S1160.hdr'
    # file='/home/shctrl/yny_intern/image_conmpression/CVQN-master/image/HDR/0801x2.png'
    # img=Image.open(file).convert('RGB')
    img=imageio.imread(file,format='HDR-FI')

    # img=Image.open(file)
    print(img)
    plt.imshow(img);
    plt.show()
    print(img.shape)

    hdr=torch.Tensor(img.transpose((2, 0, 1)))
    transform=transforms.CenterCrop((400,400))
    # hdr = transform(hdr)
    
    print(hdr.size())



    
    # a=np.array([[1,2,3],[2,3,4]])
    print(img[:,:,:].max())
    # print(torch.tensor(img))
    # transform=transforms.Compose([
    #         transforms.ToTensor(),])
    # img = transform(img)  
    # print(img)
    # print('image size:{}'.format(img.size()))

if __name__ == '__main__':
    read_img()