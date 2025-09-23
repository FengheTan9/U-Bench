from dataloader.dataset import MedicalDataSets,Covid19CTScanDataset,KvasirSEGDataset,DataScienceBowl2018Dataset,PH2Dataset,MedicalDataSetsVal,MonuSeg2018Dataset
from dataloader.dataset import KvasirSEGDatasetVAL,DRIVEdataset,CHASEDB1Dataset, BUSBRADatasets,GlasDataSets
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize
from torch.utils.data import DataLoader
from albumentations.augmentations import transforms
from dataloader.dataset_synapse import Synapse_dataset,RandomGenerator_synapse
from dataloader.dataset_ACDC import ACDCdataset,RandomGenerator_ACDC    
from dataloader.dataset_XRay import MontgomeryXRAYDataSet,MIHXRAYDataSet

from dataloader.download import get_MedSegBench_dataset 
from dataloader.download import INFO as MedSegBench_dataset_name_dict

def getDataloader(args):


    if args.model in  ["SwinUnet","SegFormer", "MissFormer", "EffiSegNetBN","HiFormer","BEFUnet", "FAT_Net", "SCUNet_plus_plus"]:
        args.img_size=224

    img_size = args.img_size

    train_transform = Compose([
        RandomRotate90(),
        transforms.Flip(),
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])
    
    if "COVID" in args.base_dir: # for COVID19;
        db_train = Covid19CTScanDataset(dataset_dir=args.base_dir, mode="train", transform=train_transform)
        db_val = Covid19CTScanDataset(dataset_dir=args.base_dir, mode="test", transform=val_transform)
    elif "Kvasir" in args.base_dir:
        dataset = KvasirSEGDataset(batch_size=args.batch_size, img_size=img_size)
        dataset.setup()
        db_train,db_val,test_set=dataset.train_set,dataset.val_set,dataset.test_set
    elif "DRIVE" in args.base_dir:
        db_train = DRIVEdataset(base_dir=args.base_dir, mode="train", transform=train_transform)
        db_val = DRIVEdataset(base_dir=args.base_dir, mode="val", transform=val_transform)
    elif "CHASEDB1" in args.base_dir:
        db_train = CHASEDB1Dataset(base_dir=args.base_dir, mode="train", transform=train_transform)
        db_val = CHASEDB1Dataset(base_dir=args.base_dir, mode="val", transform=val_transform)
        db_test = CHASEDB1Dataset(base_dir=args.base_dir, mode="test", transform=val_transform)
    elif args.dataset_name in ["bus","busi","isic18","tuscui"]: # for bus;busi;isic18;
        db_train = MedicalDataSets(base_dir=args.base_dir, mode="train", transform=train_transform,
                                train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
        db_val = MedicalDataSets(base_dir=args.base_dir, mode="val", transform=val_transform,
                                train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
    elif 'synapse' in args.base_dir:
        assert args.num_classes == 9
        db_train = Synapse_dataset(base_dir=args.base_dir, split="train", nclass=args.num_classes,
                               transform=Compose(
                                   [RandomGenerator_synapse(output_size=[args.img_size, args.img_size])]))
        db_val = Synapse_dataset(base_dir=args.base_dir, split="test_vol", nclass=args.num_classes)
    elif "ACDC" in args.base_dir:
        import torchvision
        # donot use val ；use test
        db_train = ACDCdataset(base_dir=args.base_dir, split="train", transform=torchvision.transforms.Compose([RandomGenerator_ACDC(output_size=[args.img_size, args.img_size])]))
        _ = ACDCdataset(base_dir=args.base_dir, split="valid")
        db_val = ACDCdataset(base_dir=args.base_dir, split="test")
    elif 'DSB2018' in args.base_dir:
        db_train = DataScienceBowl2018Dataset(args.base_dir, image_size=img_size, mode='train')
        db_val = DataScienceBowl2018Dataset(args.base_dir, image_size=img_size, mode='val')
        db_test = DataScienceBowl2018Dataset(args.base_dir, image_size=img_size, mode='test')
    elif 'BUSBRA' in args.base_dir:
        print("in BUSBRA")
        db_train = BUSBRADatasets(base_dir=args.base_dir, mode="train", transform=train_transform,
                                train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
        db_val = BUSBRADatasets(base_dir=args.base_dir, mode="val", transform=val_transform,
                                train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
    elif 'Glas' in args.base_dir:
        db_train = GlasDataSets(base_dir=args.base_dir, mode="train", transform=train_transform,
                                train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
        db_val = GlasDataSets(base_dir=args.base_dir, mode="val", transform=val_transform,
                                train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
    elif 'Montgomery' in args.base_dir:
        db_train = MontgomeryXRAYDataSet(base_dir=args.base_dir, mode="train", transform=train_transform,
                                train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
        db_val = MontgomeryXRAYDataSet(base_dir=args.base_dir, mode="val", transform=val_transform,
                                train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)

    elif args.dataset_name in MedSegBench_dataset_name_dict.keys():
        db_train = get_MedSegBench_dataset(flag=args.dataset_name, split="train", transform=train_transform,size=img_size)
        db_val = get_MedSegBench_dataset(flag=args.dataset_name, split="val", transform=val_transform,size=img_size)
    
    else:
        print(" data error  \n\n\n")
        exit()
        return 0
    print(f"train num:{len(db_train)}, val num:{len(db_val)}")
    import random
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,num_workers=4, pin_memory=False,worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,num_workers=1)
    return trainloader, valloader


def getZeroShotDataloader(args):

    if args.model in  ["SwinUnet","SegFormer", "MissFormer", "EffiSegNetBN","HiFormer","BEFUnet", "FAT_Net", "SCUNet_plus_plus"]:
        args.img_size=224

    img_size = args.img_size
    val_transform = Compose([
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])
    if args.zero_shot_dataset_name in ["busi","isic18","tuscui","bus","Benign","malignant", "stare"]:
        db_val = MedicalDataSetsVal(base_dir=args.zero_shot_base_dir, transform=val_transform,val_file_dir=args.val_file_dir)
    elif 'PH2' in args.zero_shot_base_dir:  # ./data/PH2Dataset/PH2
        db_val = PH2Dataset(args.zero_shot_base_dir, mode='test', transform=val_transform)
    elif "CHASEDB1" in args.zero_shot_base_dir:
        db_val = CHASEDB1Dataset(base_dir=args.zero_shot_base_dir, mode="test", transform=val_transform)
    elif 'Kvasir' in args.zero_shot_base_dir :
        dataset = KvasirSEGDatasetVAL(base_dir=args.zero_shot_base_dir,val_file_dir=args.val_file_dir,img_size=img_size)
        dataset.setup()
        db_val=dataset.val_set
    elif 'COVID19-2' in args.zero_shot_base_dir:
        db_val = Covid19CTScanDataset(args.zero_shot_base_dir, mode='test', transform=val_transform)
    elif "MonuSeg2018" in args.zero_shot_base_dir:
        db_val = MonuSeg2018Dataset(args.zero_shot_base_dir, mode='test',image_size=img_size)
    elif 'BUSBRA' in args.zero_shot_base_dir:
        db_val = BUSBRADatasets(base_dir=args.zero_shot_base_dir, mode="val", transform=val_transform,
                                train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
    elif 'Glas' in args.zero_shot_base_dir:
        db_val = GlasDataSets(base_dir=args.zero_shot_base_dir, mode="val", transform=val_transform,
                                train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
    elif 'DRIVE' in args.zero_shot_base_dir:
        db_val = DRIVEdataset(base_dir=args.zero_shot_base_dir, mode="val", transform=val_transform)
    elif 'NIH' in args.zero_shot_base_dir:
        db_val = MIHXRAYDataSet(base_dir=args.zero_shot_base_dir, mode="val", transform=val_transform)
    elif args.zero_shot_dataset_name in MedSegBench_dataset_name_dict.keys():
        db_val = get_MedSegBench_dataset(flag=args.zero_shot_dataset_name, split="test", transform=val_transform,size=img_size)
    else:
        print(f"zero shot data error {args.zero_shot_base_dir} \n\n\n")
        exit()
        return 0
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,num_workers=1)
    return valloader
