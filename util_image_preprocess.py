import numpy as np

def downsize(img_arr):
    "Downsize the frame to half the width and height"
    return img_arr[::2, ::2]

def rgb2grey(img_arr):
    "Make frame into grey scale"
    return np.true_divide(np.sum(img_arr, axis=-1), 3).astype(np.uint8)

def normalize(img):
    "Normalize pixels to values between 0 and 1"
    return np.array(img/255.0, dtype=np.float32)
    
def preprocess(img_arr):
    """ Preprecess a frame 
    Downsize, convert to grey scale and normalize pixels
    """
    grey = rgb2grey(img_arr)
    downsized = downsize(grey)
    return normalize(downsized)
