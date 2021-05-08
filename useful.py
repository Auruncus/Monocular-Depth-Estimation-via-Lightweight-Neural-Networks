import matplotlib.pyplot as plt
from zipfile import ZipFile
from sklearn.utils import shuffle
import torch
import numpy as np

def normalize_depth(depth, maxDepth=1000.0): 
    return maxDepth / depth

def load_zip(zip_file, test = False):
    print('Loading dataset zip file...', end='')
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_train = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))
    nyu2_test = list((row.split(',') for row in (data['data/nyu2_test.csv']).decode("utf-8").split('\n') if len(row) > 0))
    nyu2_train = shuffle(nyu2_train, random_state=0)

    if test: nyu2_train = nyu2_train[:40]
    if test: nyu2_test = nyu2_test[:40]

    print('Loaded ({0}).'.format(len(nyu2_train)))
    return data, nyu2_train, nyu2_test

def plot_image_depth(image, depth):
    fig, axes = plt.subplots(nrows = 1, ncols = 2,figsize=(15,15))
    axes[0].set(title='Image') 
    axes[0].imshow(image)  
    axes[1].set(title='Depth')
    axes[1].imshow(depth)  
    plt.show()
