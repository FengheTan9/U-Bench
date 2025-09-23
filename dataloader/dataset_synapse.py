
import os
import random
import h5py
import numpy as np
import torch
import cv2
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator_synapse(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image, mask, force_apply=False, **kwargs):
        if random.random() > 0.5:
            image, mask = random_rot_flip(image, mask)
        elif random.random() > 0.5:
            image, mask = random_rotate(image, mask)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            mask = zoom(mask, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(mask.astype(np.float32)).long()

        return {'image': image, 'mask': mask}


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, split, nclass=9, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        list_dir=base_dir+'/lists/lists_Synapse'
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        if split=="test_vol":
            self.data_dir = base_dir+'/test_vol_h5_new/'
        elif split=="train":
            self.data_dir = base_dir+'/train_npz_new/'
        else:
            self.data_dir = base_dir
            print("error: split must be train or test_vol")
        self.nclass = nclass

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            
            
            
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
           

        if self.nclass == 9:
            label[label==5]= 0
            label[label==9]= 0
            label[label==10]= 0
            label[label==12]= 0
            label[label==13]= 0
            label[label==11]= 5
        if self.transform:    
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']   
            image = image.repeat(3, 1, 1)
        else:
            print(image.shape)
        sample = {'image': image, 'label': label}
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample