import matplotlib.pyplot as plt
import numpy as np
from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames

def preprocess_frame(frame):
    # Greyscale frame
    gray = rgb2gray(frame)
    gray = gray[500:1500,0:1000]
    # Resize & normalize
    preprocessed_frame = transform.resize(gray, [60,60])
    
    return preprocessed_frame # 60x60 frame

def state(impath):
    im = plt.imread(impath)
    processed_im = preprocess_frame(im)
    return processed_im

