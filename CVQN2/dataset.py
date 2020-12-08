# modified from https://github.com/desimone/vision/blob/fb74c76d09bcc2594159613d5bdadd7d4697bb11/torchvision/datasets/folder.py

import os
import os.path

import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import pdb
import math
import imageio
import numpy as np

IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
    '.hdr',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    # return Image.open(path).convert('RGB')
    return imageio.imread(path)


class ImageFolder(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, root, transform=None, loader=default_loader):
        images = []
        
        for filename in os.listdir(root):
            if is_image_file(filename):
                images.append('{}'.format(filename))

        self.root = root
        self.imgs = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        # print('this is filename:{}'.format(index))
        filename = self.imgs[index]
        
        try:
          
            img = self.loader(os.path.join(self.root, filename))
            
        
        except:
            return torch.zeros((3, 32, 32))

        # print(img.size[0])
        # print(img.size[1])
        img=torch.Tensor(img.transpose((2, 0, 1)))

        # print('img size:{}'.format(img.size()))

        # if self.root=='image/HDR_MATLAB_2':
        #     h=img.size()[1]
        #     w=img.size()[2]
        #     h0=16*math.floor(h/16)
        #     w0=16*math.floor(w/16)
        #     self.transform=transforms.CenterCrop((h0,w0))

            # print('eva_img size:{}'.format(img.size()))


        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)
