import numpy as np

def RGBbt_709_np(input, peak_lum=110):
    black = peak_lum /1000.0
    lumin = (peak_lum - black) * (input/255.0)**2.2 + black
    return lumin

'''
def display_model(input, peak_lum, display='RGBbt_709'):
    if display == 'RGBbt_709':
        return RGBbt_709_np(input, peak_lum)
'''
def display_model_torch(input, peak_lum=110):
    return RGBbt_709_np(input, peak_lum=peak_lum)


def display_model(input_tup, peak_lum_tup, display='RGBbt_709'):

    physic_lum = []

    for input, peak_lum in zip(input_tup, peak_lum_tup):
        if display == 'RGBbt_709':
            physic_lum.append(RGBbt_709_np(input, peak_lum))

    return tuple(physic_lum)


#test code
if __name__ == '__main__':
    a = 255.*np.ones((512, 100, 100 ,3))
    #print(a.shape)
    #batch_xs_ref: <class 'numpy.ndarray'> (48, 48, 3)
    #batch_lumin: <class 'numpy.ndarray'> (48,)

    batch_xs = 255.*np.ones((48,48,3))
    batch
    a = display_model(a, peak_lum=110)
    print(a)
