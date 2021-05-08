import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from io import BytesIO
import random
import scipy.ndimage as ndimage
class DepthDataset(Dataset):
    def __init__(self, data, nyu2_train, transform=None):
        self.data, self.nyu_dataset = data, nyu2_train
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx]
        image = Image.open( BytesIO(self.data[sample[0]]) )
        depth = Image.open( BytesIO(self.data[sample[1]]) )
        sample = {'image': image, 'depth': depth}
        if self.transform: 
          sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.nyu_dataset)


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth}


class ToTensorTrain(object):
    def __call__(self, sample, test=False):
        image, depth = sample['image'], sample['depth']
        image = np.array(image)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()/255
        depth = np.expand_dims(np.array(sample['depth']), axis=-1)
        depth = depth.transpose((2, 0, 1))
        depth = torch.from_numpy(depth).float()/255*1000
        depth = torch.clamp(depth,10,1000)
        return {'image': image, 'depth': depth}

class ToTensorTest(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = np.array(image)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()/255
        depth = np.expand_dims(np.array(sample['depth']), axis=-1)
        depth = depth.transpose((2, 0, 1))
        depth = torch.from_numpy(depth).float()/10 #/10000 * 1000
        #depth = torch.clamp(depth,10,1000)
        return {'image': image, 'depth': depth}

class RandomRotate(object):
    def __init__(self, angle, diff_angle=0, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        applied_angle = random.uniform(-self.angle, self.angle)
        angle1 = applied_angle
        angle1_rad = angle1 * np.pi / 180
        image = ndimage.interpolation.rotate(
            image, angle1, reshape=self.reshape, order=self.order)
        depth = ndimage.interpolation.rotate(
            depth, angle1, reshape=self.reshape, order=self.order)
        image = Image.fromarray(image)
        depth = Image.fromarray(depth)

        return {'image': image, 'depth': depth}




















