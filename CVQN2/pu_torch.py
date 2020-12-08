#import tensorflow as tf
import os
import numpy as np

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
    numerator = np.log(x)
    denominator = np.log(10)
    return numerator/denominator

def pu2_encode(L):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    l_lut = np.genfromtxt(os.path.join(dir_path, 'l_lut.csv'), delimiter=',')
    P_lut = np.genfromtxt(os.path.join(dir_path, 'P_lut.csv'), delimiter=',')
    
    l_min = -5.0
    l_max = 10.0
    pu_l = 31.9270
    pu_h = 149.9244
    N = 8192.0
    
    l = log10(  np.maximum(np.minimum(L, np.power(10.0, l_max)), np.power(10.0, l_min)) )

    index = (l-l_min)*N/(l_max-l_min)
    index = np.floor(index)
    #index = tf.cast(index, tf.int32)
    print(index)
    exit()
    P_lut_t = np.stack(P_lut)
    
    encoded = 255.0*(np.gather(P_lut_t, index) - pu_l) / (pu_h-pu_l)
   
    #encoded = tf.cast(encoded, tf.float32)

    return encoded

l = 100
pu2_encode(l)

