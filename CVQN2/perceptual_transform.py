#import tensorflow as tf
import os
import numpy as np
import torch
from torchvision import transforms
import torch.utils.data as data

def pq2_encode(L):
    L_max = 10000
    hdrin = L/L_max
    #hdrin = tf.cast(hdrin, tf.float32)
    
    nVal = 0.15930175781250000;
    mVal = 78.84375;
    c1val = 0.8359375;
    c2val = 18.8515625;
    c3val = 18.6875;
    temp = np.power(hdrin, nVal)
    encoded = np.power((c2val*temp + c1val)/(1+c3val*temp), mVal)
    #encoded = tf.cast(encoded, tf.float32)
    return encoded


def log10(x):
    numerator = torch.log(x).cuda()
    denominator = torch.log(torch.Tensor([10,])).cuda()
    return numerator/denominator

def pu2_encode(L):
    batch_size=L.size()[0]
    h_size=L.size()[2]
    w_size=L.size()[3]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    l_lut = np.genfromtxt(os.path.join(dir_path, 'l_lut.csv'), delimiter=',')
    P_lut = np.genfromtxt(os.path.join(dir_path, 'P_lut.csv'), delimiter=',')
    
    l_min = -5.0
    l_max = 10.0
    pu_l = 31.9270
    pu_h = 149.9244
    N = 8192.0



    l = log10(  torch.max(torch.minimum(L.cuda(),torch.pow(torch.Tensor([10.0,]).cuda(), l_max).cuda()).cuda(), torch.pow(torch.Tensor([10.0,]).cuda(), l_min).cuda()).cuda() )

    index = (l-l_min)*N/(l_max-l_min)
    index=index.floor().long()

    index=index.view(-1,1)


    P_lut_t = torch.Tensor(P_lut).view(-1,1)

    encoded = 255.0*(torch.index_select(P_lut_t.cuda(),0,torch.squeeze(index).cuda()) - pu_l) / (pu_h-pu_l)

    encoded=encoded.view(batch_size,3,h_size,w_size)
   
    #encoded = tf.cast(encoded, tf.float32)
    
    return encoded


# ## =============test==================

# l = 10000.*np.ones((1,3,2,2))

# img=torch.Tensor(l)

# print(img.size())

# img[0,1,0,0]=1
# img[0,1,0,1]=10
# img[0,1,1,0]=100
# img[0,1,1,1]=1000

# img[0,2,0,0]=10
# img[0,2,0,1]=100
# img[0,2,1,0]=1000
# img[0,2,1,1]=10000

# print(img.size())

# pu=pu2_encode(img)

# print(pu)
# print(img.shape)
# print(pu.shape)

