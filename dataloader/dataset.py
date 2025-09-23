import os
import cv2
from torch.utils.data import Dataset
# from dataloader.BioMedicalDataset.PH2Dataset import PH2Datasetmy
# from dataloader.BioMedicalDataset.Covid19CTScanDataset import Covid19CTScanDataset
import pandas as pd
import numpy as np
#import lightning as L
import albumentations as A
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from albumentations.pytorch import ToTensorV2


class MedicalDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        mode="train",
        transform=None,
        train_file_dir="train.txt",
        val_file_dir="val.txt",
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.mode = mode
        self.transform = transform
        self.train_list = []
        self.semi_list = []

        if self.mode == "train":
            with open(os.path.join(self._base_dir, train_file_dir), "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.mode == "val":
            with open(os.path.join(self._base_dir, val_file_dir), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("total {}  {} samples".format(len(self.sample_list), self.mode))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        case = self.sample_list[idx]

        image = cv2.imread(os.path.join(self._base_dir, 'images', case + '.png'))
        label = cv2.imread(os.path.join(self._base_dir, 'masks', '0', case + '.png'), cv2.IMREAD_GRAYSCALE)[..., None]
        augmented = self.transform(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']

        image = image.astype('float32')
        image = image.transpose(2, 0, 1) / 255

        label = label.astype('float32') 
        label = label.transpose(2, 0, 1) / 255

        label[label > 0] = 1

        sample = {"image": image, "label": label, "case": case}
        return sample



class GlasDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        mode="train",
        transform=None,
        train_file_dir="train.txt",
        val_file_dir="val.txt",
    ):
        
        self._base_dir = base_dir
        self.sample_list = []
        self.mode = mode
        self.transform = transform
        self.train_list = []
        self.semi_list = []

        if self.mode == "train":
            with open(os.path.join(self._base_dir,train_file_dir), "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.mode == "val":
            with open(os.path.join(self._base_dir, val_file_dir), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("total {}  {} samples".format(len(self.sample_list), self.mode))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        case = self.sample_list[idx]
        
        image = cv2.imread(os.path.join(self._base_dir+'/glas_dataset/' ,case))
        label = cv2.imread(os.path.join(self._base_dir+'/glas_dataset/' ,case.replace('.bmp','_anno.bmp')), cv2.IMREAD_GRAYSCALE)[..., None]
        augmented = self.transform(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']

        image = image.astype('float32')
        image = image.transpose(2, 0, 1) / 255

        label = label.astype('float32') 
        label = label.transpose(2, 0, 1) / 255

        label[label > 0] = 1

        sample = {"image": image, "label": label, "case": case}
        return sample




class BUSBRADatasets(Dataset):
    def __init__(
        self,
        base_dir=None,
        mode="train",
        transform=None,
        train_file_dir="train.txt",
        val_file_dir="val.txt",
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.mode = mode
        self.transform = transform
        self.train_list = []
        self.semi_list = []

        if self.mode == "train":
            with open(os.path.join(self._base_dir, train_file_dir), "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.mode == "val":
            with open(os.path.join(self._base_dir, val_file_dir), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("total {}  {} samples".format(len(self.sample_list), self.mode))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        case = self.sample_list[idx]

        image = cv2.imread(os.path.join(self._base_dir, 'Images', case ))
        label = cv2.imread(os.path.join(self._base_dir, 'Masks', case.replace('bus','mask')), cv2.IMREAD_GRAYSCALE)[..., None]
        
        augmented = self.transform(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']

        image = image.astype('float32')
        image = image.transpose(2, 0, 1) / 255

        label = label.astype('float32') 
        label = label.transpose(2, 0, 1) / 255

        label[label > 0] = 1

        sample = {"image": image, "label": label, "case": case}
        return sample



class MedicalDataSetsVal(Dataset):
    def __init__(
        self,
        base_dir=None,
        transform=None,
        val_file_dir="val.txt",
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.transform = transform
        self.train_list = []
        self.semi_list = []
        with open(os.path.join(self._base_dir, val_file_dir), "r") as f1:
            self.sample_list = f1.readlines()
        self.sample_list = [item.replace("\n", "") for item in self.sample_list]


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        case = self.sample_list[idx]

        image = cv2.imread(os.path.join(self._base_dir, 'images', case + '.png'))
        label = cv2.imread(os.path.join(self._base_dir, 'masks', '0', case + '.png'), cv2.IMREAD_GRAYSCALE)[..., None]

        augmented = self.transform(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']

        image = image.astype('float32') / 255
        image = image.transpose(2, 0, 1)

        label = label.astype('float32') / 255
        label = label.transpose(2, 0, 1)

        label[label > 0] = 1

        
        sample = {"image": image, "label": label, "case": case}
        return sample






class MedicalDataSetsVal_withscale(Dataset):
    def __init__(
        self,
        base_dir=None,
        transform=None,
        val_file_dir="val.txt",
        noise_type=None, 
        noise_level=0.1, 
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.transform = transform
        self.train_list = []
        self.semi_list = []
        self.noise_type = noise_type 
        self.noise_level = noise_level
        with open(os.path.join(self._base_dir, val_file_dir), "r") as f1:
            self.sample_list = f1.readlines()
        self.sample_list = [item.replace("\n", "") for item in self.sample_list]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image = cv2.imread(os.path.join(self._base_dir, 'images', case + '.png'))
        label = cv2.imread(os.path.join(self._base_dir, 'masks', '0', case + '.png'), cv2.IMREAD_GRAYSCALE)[..., None]
        
        if self.noise_type == 'gaussian':
            
            noise_std = min(255, self.noise_level * 255)
            
            noise = np.clip(np.random.normal(0, noise_std, image.shape), 0, 255).astype(np.int16)
            image = cv2.add(image, noise.astype(np.uint8))
        elif self.noise_type == 'salt_pepper':
            noise_mask = np.random.choice([0, 1, 2], size=image.shape[:2], 
                                        p=[1-self.noise_level, self.noise_level/2, self.noise_level/2])
            image[noise_mask == 1] = 255  
            image[noise_mask == 2] = 0    

        augmented = self.transform(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']

        foreground_pixels = (label > 10).sum()
        ratio = foreground_pixels / (label.shape[0]*label.shape[1] * label.shape[2])

        image = image.astype('float32') / 255
        image = image.transpose(2, 0, 1)

        label = label.astype('float32') / 255
        label = label.transpose(2, 0, 1)

        sample = {"image": image, "label": label, "case": case, 'ratio': ratio}
        return sample




class KvasirSEGDataset(Dataset):
    def __init__(
        self,
        batch_size=16,
        root_dir="./data/Kvasir-SEG",
        num_workers=2,
        train_val_ratio=0.8,
        img_size=256,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.num_workers = num_workers
        self.train_val_ratio = train_val_ratio
        self.img_size = img_size

    def get_train_transforms(self):
        return A.Compose(
            [
                A.Resize(*(self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(
                    p=0.5, brightness=(0.6, 1.6), contrast=0.2, saturation=0.1, hue=0.01
                ),
                A.Affine(
                    p=0.5,
                    scale=(0.5, 1.5),
                    translate_percent=0.125,
                    rotate=90,
                    interpolation=cv2.INTER_LANCZOS4,
                ),
                A.ElasticTransform(p=0.5, interpolation=cv2.INTER_LANCZOS4),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255,
                ),
                ToTensorV2(),
            ]
        )

    def get_val_transforms(self):
        return A.Compose(
            [
                A.Resize(*(self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255,
                ),
                ToTensorV2(),
            ]
        )

    def get_test_transforms(self):
        return A.Compose(
            [
                A.Resize(*(self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255,
                ),
                ToTensorV2(),
            ]
        )

    def setup(self, stage=None):
        train_images = os.listdir(os.path.join(self.root_dir, "train/images"))
        train_masks = os.listdir(os.path.join(self.root_dir, "train/masks"))
        train_images = [os.path.join(self.root_dir, "train/images", img) for img in train_images]
        train_masks = [os.path.join(self.root_dir, "train/masks", mask) for mask in train_masks]

        val_images = os.listdir(os.path.join(self.root_dir, "validation/images"))
        val_masks = os.listdir(os.path.join(self.root_dir, "validation/masks"))
        val_images = [os.path.join(self.root_dir, "validation/images", img) for img in val_images]
        val_masks = [os.path.join(self.root_dir, "validation/masks", mask) for mask in val_masks]

        test_images = os.listdir(os.path.join(self.root_dir, "test/images"))
        test_masks = os.listdir(os.path.join(self.root_dir, "test/masks"))
        test_images = [os.path.join(self.root_dir, "test/images", img) for img in test_images]
        test_masks = [os.path.join(self.root_dir, "test/masks", mask) for mask in test_masks]

        train_pairs = list(zip(train_images, train_masks))
        val_pairs = list(zip(val_images, val_masks))
        test_pairs = list(zip(test_images, test_masks))

        self.train_set = KvasirSEGDatagen(
            train_pairs, transform=self.get_train_transforms()
        )
        self.val_set = KvasirSEGDatagen(val_pairs, transform=self.get_val_transforms())
        self.test_set = KvasirSEGDatagen(test_pairs, transform=self.get_test_transforms())

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class PH2Dataset(Dataset) :
    def __init__(self, dataset_dir, mode, transform=None):
        super(PH2Dataset, self).__init__()
        self.image_folder = 'images'
        self.label_folder = 'masks'
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.frame = pd.read_csv(os.path.join(dataset_dir, '{}_frame.csv'.format(mode)))

        print("self.frame",len(self.frame))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_dir, self.image_folder, self.frame.image_path[idx])
        label_path = os.path.join(self.dataset_dir, self.label_folder, self.frame.mask_path[idx])


        image = cv2.imread(image_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)[..., None]

        if self.transform:        
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']

        image = image.astype('float32') / 255
        image = image.transpose(2, 0, 1)

        label = label.astype('float32') / 255
        label = label.transpose(2, 0, 1)
        label[label >= 0.5] = 1; label[label < 0.5] = 0
        sample = {"image": image, "label": label, "case": self.frame.image_path[idx]}
        return sample



class KvasirSEGDatagen(Dataset):
    def __init__(self, pairs, transform=None):
        self.transform = transform
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image = cv2.imread(self.pairs[idx][0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.pairs[idx][1], 0)
        mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)[1].astype('float32')


        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        mask=mask.unsqueeze(0)
        sample = {"image": image, "label": mask, "case": self.pairs[idx][0]}

        return sample



class KvasirSEGDatagenVAL(Dataset):
    def __init__(self, pairs, transform=None):
        self.transform = transform
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image = cv2.imread(self.pairs[idx][0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.pairs[idx][1], 0)
        mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)[1].astype('float32') 

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        mask=mask.unsqueeze(0)
        sample = {"image": image, "label": mask, "case": self.pairs[idx][0]}

        return sample


class KvasirSEGDatasetVAL(Dataset):
    def __init__(
        self,
        base_dir=None,
        val_file_dir="val.txt",
        img_size=256,
    ):
        super().__init__()
        self._base_dir = base_dir
        self.val_file_dir=val_file_dir
        self.img_size = img_size


        with open(os.path.join(self._base_dir, val_file_dir), "r") as f:
            self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

    def get_val_transforms(self):
        return A.Compose(
            [
                A.Resize(*(self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255,
                ),
                ToTensorV2(),
            ]
        )

    def setup(self, stage=None):


        val_images = [os.path.join(self._base_dir, img + '.png') for img in self.sample_list]
        val_masks = [os.path.join(self._base_dir,  img.replace("images","masks") + '.png') for img in self.sample_list]

        val_pairs = list(zip(val_images, val_masks))
        self.val_set = KvasirSEGDatagenVAL(val_pairs, transform=self.get_val_transforms())

  
    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            shuffle=False,
        )






class CHASEDB1Dataset(Dataset):
    def __init__(
        self,
        base_dir=None,
        mode="train",
        transform=None,
        train_file_dir="train.txt",
        val_file_dir="val.txt",
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.mode = mode
        self.transform = transform
        self.train_list = []
        self.semi_list = []

        if self.mode == "train":
            image_dir = os.path.join(self._base_dir, "train", "images")
            label_dir = os.path.join(self._base_dir, "train", "labels")
        elif self.mode == "val":
            image_dir = os.path.join(self._base_dir, "val", "images")
            label_dir = os.path.join(self._base_dir, "val", "labels")
        else:
            image_dir = os.path.join(self._base_dir, "test", "images")
            label_dir = os.path.join(self._base_dir, "test", "labels")
        image_files = sorted(os.listdir(image_dir))
        label_files = sorted(os.listdir(label_dir))

        assert len(image_files) == len(label_files), "Number of images and labels must match"

        self.sample_list = [
            (os.path.join(image_dir, img_file), os.path.join(label_dir, label_file))
            for img_file, label_file in zip(image_files, label_files)
        ]

        print("total {}  {} samples".format(len(self.sample_list), self.mode))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        image_path, label_path = self.sample_list[idx]

        image = cv2.imread(image_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)[..., None]

        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']

        image = image.astype('float32')  / 255
        image = image.transpose(2, 0, 1)

        label = label.astype('float32') / 255
        label = label.transpose(2, 0, 1)

        sample = {"image": image, "label": label, "case": os.path.basename(image_path)}
        return sample



class DRIVEdataset(Dataset):
    def __init__(
        self,
        base_dir=None,
        mode="train",
        transform=None
    ):
        self._base_dir = base_dir
        self.mode = mode
        self.transform = transform

        if self.mode == "train":
            folder_input = os.path.join(self._base_dir, "train_data/images")
            folder_label = os.path.join(self._base_dir, "train_data/1st_manual")
        elif self.mode == "test" or self.mode == "val":
            folder_input = os.path.join(self._base_dir, "test_data/images")
            folder_label = os.path.join(self._base_dir, "test_data/1st_manual")
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        input_list = os.listdir(folder_input)
        label_list = os.listdir(folder_label)

        input_list = sorted(input_list)
        label_list = sorted(label_list)

        self.input_list = [os.path.join(folder_input, name) for name in input_list]
        self.label_list = [os.path.join(folder_label, name) for name in label_list]

        print("total {}  {} samples".format(len(self.input_list), self.mode))

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        img_path = self.input_list[idx]
        label_path = self.label_list[idx]

        img = Image.open(img_path)
        label = Image.open(label_path)
        img =   np.array(img)
        label = np.array(label)[..., None]

        if self.transform:
            augmented = self.transform(image=img, mask=label)
            image = augmented['image']
            label = augmented['mask']
        image = image.astype('float32')  / 255
        image = image.transpose(2, 0, 1)
        label = label.astype('float32') 
        label = label.transpose(2, 0, 1) / 255
        sample = {"image": image, "label": label,"case":self.input_list[idx]}
        return sample




class Covid19CTScanDataset(Dataset) :
    def __init__(self, dataset_dir, mode, transform=None, target_transform=None):
        super(Covid19CTScanDataset, self).__init__()
        self.image_folder = 'images'
        self.label_folder = 'masks'
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.target_transform = target_transform
        self.frame = pd.read_csv(os.path.join(f"{dataset_dir}", '{}_frame.csv'.format(mode)))#[:10]

        print(len(self.frame))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_dir,  self.image_folder, self.frame.image_path[idx])
        label_path = os.path.join(self.dataset_dir,  self.label_folder, self.frame.mask_path[idx])

        image = cv2.imread((image_path))
        label = cv2.imread((label_path), cv2.IMREAD_GRAYSCALE)[..., None]

        augmented = self.transform(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']

        image = image.astype('float32') / 255
        image = image.transpose(2, 0, 1)
        label = label.astype('float32') / 255
        label = label.transpose(2, 0, 1)
        sample = {"image": image, "label": label, "case": self.frame.image_path[idx]}
        return sample
    

import random
import torch
class MonuSeg2018Dataset(Dataset): 
    def __init__(self, dataset_dir, mode,image_size):
        super(MonuSeg2018Dataset, self).__init__()

        self.image_folder = 'images'
        self.label_folder = 'masks'

        self.dataset_dir = dataset_dir
        self.frame = pd.read_csv(os.path.join(dataset_dir, '{}_frame.csv'.format(mode)))
        self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                ])

        self.target_transform =  transforms.Compose([
                    transforms.Resize((image_size,image_size)),
                    transforms.ToTensor(),
                ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_dir, self.image_folder, self.frame.image_path[idx])
        label_path = os.path.join(self.dataset_dir, self.label_folder, self.frame.mask_path[idx])

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.transform:
            seed = random.randint(0, 2 ** 32)
            self._set_seed(seed); image = self.transform(image)
            self._set_seed(seed); label = self.target_transform(label)

        label[label >= 0.5] = 1; label[label < 0.5] = 0
        sample = {"image": image, "label": label, "case": self.frame.image_path[idx]}

        return sample

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)




class DataScienceBowl2018Dataset(Dataset) : 
    def __init__(self, dataset_dir, mode,image_size, transform=None, target_transform=None):
        super(DataScienceBowl2018Dataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.image_folder = 'stage1_train'
        self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                ])

        self.target_transform =  transforms.Compose([
                    transforms.Resize((image_size,image_size)),
                    transforms.ToTensor(),
                ])

        self.frame = pd.read_csv(os.path.join(dataset_dir, '{}_imageid_frame.csv'.format(mode)))


    def __len__(self):
        return len(self.frame.ImageId.unique())

    def _get_image(self, idx):
        img_name = self.frame.ImageId.unique()[idx]
        img_path = os.path.join(self.dataset_dir, self.image_folder, img_name, "images", img_name + '.png')

        image = Image.open(img_path).convert('RGB')

        return image

    def _load_mask(self, idx): 
        img_name = self.frame.ImageId.unique()[idx]
        mask_dir = os.path.join(self.dataset_dir, self.image_folder, img_name, "masks")
        mask_paths = [os.path.join(mask_dir, fp) for fp in os.listdir(mask_dir)]
        mask = None
        for fp in mask_paths:
            img = cv2.imread(fp, 0)
            if img is None:
                raise FileNotFoundError("Could not open %s" % fp)
            if mask is None:
                mask = img
            else:
                mask = np.maximum(mask, img)

        mask = Image.fromarray(mask)

        return mask

    def __getitem__(self, idx):
        image = self._get_image(idx)
        mask = self._load_mask(idx)

        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, a = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        b = np.count_nonzero(a)
        ttl = np.prod(a.shape)
        if b > ttl / 2:
            image = Image.fromarray(cv2.bitwise_not(img))

        if self.transform:
            seed = random.randint(0, 2 ** 32)
            self._set_seed(seed); image = self.transform(image)
            self._set_seed(seed); mask = self.target_transform(mask)
        mask[mask >= 0.5] = 1; mask[mask < 0.5] = 0

        if image.shape[0] == 1: image = image.repeat(3, 1, 1)

        number = len(self.frame[self.frame.ImageId == self.frame.ImageId.unique()[idx]])
        sample = {"image": image, "label": mask, "case": self.frame.ImageId.unique()[idx]}
        return sample

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)