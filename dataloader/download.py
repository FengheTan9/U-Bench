
from torch import from_numpy
from torch.utils.data import Dataset


from PIL import Image
import numpy as np
import os


import os
from os.path import expanduser
import warnings


def get_default_root():
    home = expanduser("~")
    dir_path = os.path.join(home, ".medsegbench")

    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except:
        warnings.warn("Failed to setup default root.")
        dir_path = None
    
    return dir_path

HOMEPAGE = "[URL]"

INFO = {
    'kvasir': {
        'python_class' : "KvasirMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : '2c703d693d16437a2891f1aec900a1ea',
        'url_256' : '[URL]',
        'MD5_256' : '4ce7118decc9973a252e9a9dd6c1e47e',
        'url_512' : '[URL]',
        'MD5_512' : 'f3ab71c25d07077c85b5d98cfe615f9d',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'
        },
        'n_channels_im' : 3,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 700, 'val' : 100, 'test' : 200},
        'license' : 'The use of the Kvasir-SEG dataset is restricted for research and educational purposes. The use of the Kvasir-SEG dataset for commercial purposes is forbidden without prior written permission.',
        'reference': '10.1007/978-3-030-37734-2_37, [URL]'
    },
    
    'isic2018': {
        'python_class' : "Isic2018MSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : 'a89f5259e99e146f1e983173ced69903',
        'url_256' : '[URL]',
        'MD5_256' : '03680ffc3fc8e0221f161d445c8a94d4',
        'url_512' : '[URL]',
        'MD5_512' : '000a76e688e6d48543b9563a8c406ae4',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'
        },
        'n_channels_im' : 3,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 2594, 'val' : 100, 'test' : 1000},
        'license' : 'CC BY-NC 4.0',
        'reference': '10.48550/arXiv.1902.03368, 10.1038/sdata.2018.161, [URL]'
    },    
    'drive': {
        'python_class' : "DriveMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : '39582fd6528bdd7b66ed92c6dd9f26aa',
        'url_256' : '[URL]',
        'MD5_256' : '56f96cf647df7ba48c56586c8d0d2cf8',
        'url_512' : '[URL]',
        'MD5_512' : '9182813e1f337e6207ce421847fb0b0b',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'
        },
        'n_channels_im' : 3,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 18, 'val' : 2, 'test' : 20},
        'license' : 'CC-BY-4.0',
        'reference': '10.1109/TMI.2004.825627, [URL]'
    },

    'chasedb1': {
        'python_class' : "ChaseDB1MSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : '55e391c5002493c77ceb912ef1070cbe',
        'url_256' : '[URL]',
        'MD5_256' : 'd72833a2603872150a12b2763cfe454f',
        'url_512' : '[URL]',
        'MD5_512' : '7c94b4d937474a52605cc5c46a092660',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'
        },
        'n_channels_im' : 3,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 19, 'val' : 2, 'test' : 7},
        'license' : 'CC BY 4.0',
        'reference': '10.1016/j.bspc.2018.06.007, [URL]'
    },

    'dca1': {
        'python_class' : "Dca1MSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : 'cf95012398d384bb82f53b84c8051df8',
        'url_256' : '[URL]',
        'MD5_256' : '1488e23fa62d49c6c275d634e78eac5b',
        'url_512' : '[URL]',
        'MD5_512' : '020059b0e1d0afb8b5d41d7c8e48cd78',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'
        },
        'n_channels_im' : 1,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 93, 'val' : 13, 'test' : 28},
        'license' : 'This database represents the first set of angiograms and ground-truth images made accessible to the scientific community for research and comparison purposes',
        'reference': '[URL] [URL]'
    },

    'chuac': {
        'python_class' : "ChuacMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : '8d3c9e695209c3268828b7e91dac8fed',
        'url_256' : '[URL]',
        'MD5_256' : '0d968dc7adf83f5f7348b88b807dbf14',
        'url_512' : '[URL]',
        'MD5_512' : 'e5dffcc711859e3b67fc9cfefe4f1372',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'
        },
        'n_channels_im' : 1,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 21, 'val' : 3, 'test' : 6},
        'license' : 'CC BY 4.0',
        'reference': '[URL]'
    },

    'wbc': {
        'python_class' : "WbcMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : '7336757d11cec155d3cb940d3ab92372',
        'url_256' : '[URL]',
        'MD5_256' : '5002dfde36536a9f04eeeae87e0631ec',
        'url_512' : '[URL]',
        'MD5_512' : '30d13d8851d27731cdfb51e64cc4ea6b',
        'task'    : 'multi-class',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'cytoplasm',
            '2' : 'nucleus'
        },
        'n_channels_im' : 3,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 280, 'val' : 40, 'test' : 80,
                       'train_C1' : 146, 'val_C1' : 20, 'test_C1' : 43,
                       'train_C2' : 63, 'val_C2' : 9, 'test_C2' : 18,
                       'train_C3' : 44, 'val_C3' : 6, 'test_C3' : 13,
                       'train_C4' : 23, 'val_C4' : 3, 'test_C4' : 8,},
        'license' : 'CC BY 4.0',
        'class_names' : {"C1" : "Lymphocyte", "C2" : "Monocyte", "C3": "Neutrophil", "C4": "Eosinophil"},
        'reference': '10.1016/j.micron.2018.01.010, 10.1016/j.cmpb.2019.105020'
    },

    'idrib': {
        'python_class' : "IdribMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : 'f008052164fa4cc79b34df984a0e3eac',
        'url_256' : '[URL]',
        'MD5_256' : '543c16d496b843655b65df6128474275',
        'url_512' : '[URL]',
        'MD5_512' : 'ed2f6550e8bb7b9b0ab82c457d9a30be',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'            
        },
        'n_channels_im' : 3,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 47, 'val' : 6, 'test' : 27,
                       'train_C1': 47, 'val_C1': 6, 'test_C1': 27,
                       'train_C2': 47, 'val_C2': 6, 'test_C2': 27,
                       'train_C3': 47, 'val_C3': 6, 'test_C3': 27,
                       'train_C4': 47, 'val_C4': 6, 'test_C4': 27},
        'license' : 'CC BY 4.0',
        'class_names' : {"C1" : "Microaneurysms", "C2" : "Hemorrhages", "C3": "Hard Exudates", "C4": "Optic Disc"},
        'reference': '10.3390/data3030025, 10.21227/H25W98'
    },

    'cellnuclei': {
        'python_class' : "CellnucleiMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : '65440774822a5d13e015915661cc9f6c',
        'url_256' : '[URL]',
        'MD5_256' : 'bce679a41ed55486a0a5b0a7471f2b43',
        'url_512' : '[URL]',
        'MD5_512' : '3fca3b448eb4985343d80fe0819ac4d6',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'            
        },
        'n_channels_im' : 3,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 469, 'val' : 67, 'test' : 134},
        'license' : 'CC0',
        'reference': '10.1038/s41592-019-0612-7, [URL]'
    },

    'pandental': {
        'python_class' : "PandentalMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : '08a455cb39ddd0e80ac79c01a631fd4c',
        'url_256' : '[URL]',
        'MD5_256' : 'bbf9aecae003ab868958c8bb66fdf534',
        'url_512' : '[URL]',
        'MD5_512' : '6387bfe558ea9cabab80460ddb4f8112',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            'non-zero' : 'foreground'            
        },
        'n_channels_im' : 1,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 81, 'val' : 11, 'test' : 24},
        'license' : 'CC BY 4.0',
        'reference' : '10.1117/1.JMI.2.4.044003, 10.17632/hxt48yk462.1'
    },

    'bbbc010': {
        'python_class' : "Bbbc010MSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : 'ee120b3e524d0e545cd0ef14a3a51769',
        'url_256' : '[URL]',
        'MD5_256' : 'b4c4e7efab9cc6106557bf8925203203',
        'url_512' : '[URL]',
        'MD5_512' : 'b13cb2a52ff3705e8df1bd88cd1a7d54',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'            
        },
        'n_channels_im' : 1,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 70, 'val' : 10, 'test' : 20},
        'license' : 'Does not have copyright information',
        'reference': '[URL] 10.1038/nmeth.2083'
    },

    'dynamicnuclear': {
        'python_class' : "DynamicNuclearMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : 'd03dd5f0111572ded704088547cdd343',
        'url_256' : '[URL]',
        'MD5_256' : '23f77cfaa57f97379041cf5642d3ea7a',
        'url_512' : '[URL]',
        'MD5_512' : 'a9fa203eea7adfae4d6221518a9a0a76',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            'non-zero' : 'foreground'            
        },
        'n_channels_im' : 1,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 4950, 'val' : 1417, 'test' : 717},
        'license' : 'Modified Apache 2.0',
        'reference': '10.1371/journal.pcbi.1005177, [URL]'
    },

    'nuset': {
        'python_class' : "NusetMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : 'fa00c1d07f70bebfde143cdd1fb71a0d',
        'url_256' : '[URL]',
        'MD5_256' : 'adca30f019226ac54edfcf065b407d1c',
        'url_512' : '[URL]',
        'MD5_512' : '0f8721a9df92da77ec1825c7917e6886',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'            
        },
        'n_channels_im' : 1,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 2385, 'val' : 340, 'test' : 683},
        'license' : 'CC BY 4.0',
        'reference': '10.1371/journal.pcbi.1008193, 10.5281/zenodo.3996369,'
    },

    'nuclei': {
        'python_class' : "NucleiMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : 'ed5f59ec7b70a44dbe5215a9adb39963',
        'url_256' : '[URL]',
        'MD5_256' : 'b7039dba2f084063b33d89ea4ec90cb6',
        'url_512' : '[URL]',
        'MD5_512' : 'ac99716bd0d0d7ee886feb28ed5549b1',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'            
        },
        'n_channels_im' : 3,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 98, 'val' : 14, 'test' : 29},
        'license' : 'BSD-3',
        'reference': '10.4103/2153-3539.186902, [URL]'
    },

    'tnbcnuclei': {
        'python_class' : "TnbcnucleiMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : '011766acd5f140a7ad577b13b45b34ec',
        'url_256' : '[URL]',
        'MD5_256' : '5b2a8a2ef9d18971cc27c0c4529f0fcf',
        'url_512' : '[URL]',
        'MD5_512' : '92189b8e385ab625a64ce9823ac354aa',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'            
        },
        'n_channels_im' : 3,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 35, 'val' : 5, 'test' : 10},
        'license' : 'CC BY 4.0',
        'reference': '10.1109/TMI.2018.2865709, [URL]'
    },

    'monusac': {
        'python_class' : "MonusacMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : 'b8866d8ad0ed2bdcba7dd1bc82ea4e62',
        'url_256' : '[URL]',
        'MD5_256' : '4fd7301493f76c79de2fee37faa30fce',
        'url_512' : '[URL]',
        'MD5_512' : '2b75fd1ba3b91586a2b0da1594a70d29',
        'task'    : 'multi-class',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'Epithelial',
            '2' : 'Lymphocyte',
            '3' : 'Macrophage',
            '4' : 'Neutrophil',
            '5' : 'Ambiguous'          
        },
        'n_channels_im' : 3,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 188, 'val' : 21, 'test' : 101},
        'license' : 'CC BY-NC-SA 4.0',
        'reference': '10.1109/TMI.2021.3085712, [URL]'
    },

    'deepbacs': {
        'python_class' : "DeepbacsMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : '496516a356190d4d84e081224182583b',
        'url_256' : '[URL]',
        'MD5_256' : '5bd2d12a641fed6889a148cfb8ab4fd3',
        'url_512' : '[URL]',
        'MD5_512' : '75c2299a6f7e2386d3342ced7e7827ca',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'            
        },
        'n_channels_im' : 1,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 17, 'val' : 2, 'test' : 15 },
        'license' : 'CC BY 4.0',
        'reference': '10.1038/s42003-022-03634-z, 10.5281/zenodo.5550934'
    },

    'brifiseg': {
        'python_class' : "BriFiSegMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : '8ad36916999769d94a98b838226fd589',
        'url_256' : '[URL]',
        'MD5_256' : 'be00218d75aaa50624c63077e9efc843',
        'url_512' : '[URL]',
        'MD5_512' : 'aa1bda7ba872defaba2c8762af1987e7',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'            
        },
        'n_channels_im' : 1,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 1005, 'val' : 115, 'test' : 240,
                       'train_C1' : 201, 'val_C1' : 23, 'test_C1' : 48,
                       'train_C2' : 201, 'val_C2' : 23, 'test_C2' : 48,
                       'train_C3' : 201, 'val_C3' : 23, 'test_C3' : 48,
                       'train_C4' : 201, 'val_C4' : 23, 'test_C4' : 48,
                       'train_C5' : 201, 'val_C5' : 23, 'test_C5' : 48},
        'class_name': {"C1": "Target 1 A549", "C2":"Target 2 A549", "C3": "HeLa", "C4": "MCF7", "C5": "RPE1"},
        'license' : 'CC BY 4.0',
        'reference': '[URL] [URL]'
    },

    # X
    'busi': {
        'python_class' : "BusiMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : '68877751adef0ab639de73946aac0985',
        'url_256' : '[URL]',
        'MD5_256' : '198aea70968b71adf593b32c41a6e995',
        'url_512' : '[URL]',
        'MD5_512' : 'fab7550a6c5240e35da8ce5d31bc1c61',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'            
        },
        'n_channels_im' : 1,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 452, 'val' : 64, 'test' : 131,
                       'train_C1' : 305, 'val_C1' : 43, 'test_C1' : 89,
                       'train_C2' : 147, 'val_C2' : 21, 'test_C2' : 42},
        'class_names' : {"C1" : "benign", "C2" : "malignant"},
        'license' : 'CC0: Public Domain',
        'reference': '10.1016/j.dib.2019.104863, [URL]'
    },

    'covid19radio': {
        'python_class' : "Covid19RadioMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : '49c72b01477c1656acd2842ec71df5fd',
        'url_256' : '[URL]',
        'MD5_256' : 'f1b1de2f1a22d163f553d9b554c742ec',
        'url_512' : '[URL]',
        'MD5_512' : '69c39062cd60a45cd8d0ed1a1b43e493',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'            
        },
        'n_channels_im' : 1,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 14814, 'val' : 2115, 'test' : 4236,
                       'train_C1' : 2531,  'val_C1' : 361,  'test_C1' : 724,
                       'train_C2' : 4208,  'val_C2' : 601,  'test_C2' : 1203,
                       'train_C3' : 7134,  'val_C3' : 1019, 'test_C3' : 2039,
                       'train_C4' : 941,   'val_C4' : 134,  'test_C4' : 270},
        'class_names' : {"C1" : "covid", "C2" : "lung", "C3": "normal", "C4": "viral pneumonia"},
        'license' : 'Academic/Non-Commercial Use',
        'reference': '10.1109/ACCESS.2020.3010287, 10.1016/j.compbiomed.2021.104319, [URL]'
    },

    'covidquex': {
        'python_class' : "CovidQUExMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : '338a3118801d5cbcbb5986bb5178a7a2',
        'url_256' : '[URL]',
        'MD5_256' : '75861f1f34a4af16f1ab91d93b5518a0',
        'url_512' : '[URL]',
        'MD5_512' : 'ea1a64adbb57ca7a60373c1b244fcff6',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'            
        },
        'n_channels_im' : 1,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 1864, 'val' : 466, 'test' : 583 },
        'license' : 'CC BY-SA 4.0',
        'reference': '10.1016/j.compbiomed.2021.105002, 10.34740/kaggle/dsv/3122958, 10.1016/j.compbiomed.2021.104319, 10.1007/s13755-021-00146-8, 10.1109/ACCESS.2020.3010287, [URL]' 
    },
    
    'cystoidfluid': {
        'python_class' : "CystoFluidMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : 'af9e66b097c3f78d3e1a5b88efdb85cf',
        'url_256' : '[URL]',
        'MD5_256' : 'd1eadcfcc12fd560fcb52afc596422da',
        'url_512' : '[URL]',
        'MD5_512' : 'e16c60557bd7ec391b5fffb7feb9d1e1',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'            
        },
        'n_channels_im' : 3,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 703, 'val' : 101, 'test' : 202 },
        'license' : 'CC BY-NC-SA 4.0',
        'reference': '10.1109/TMI.2021.3057884, 10.34740/KAGGLE/DS/2277068, [URL]'
    },

    'isic2016': {
        'python_class' : "Isic2016MSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : '4fcee7c35a990f25f2c82ad3ed2f4f54',
        'url_256' : '[URL]',
        'MD5_256' : 'ee3fc6b5fffdc039e963ab21ff18e42e',
        'url_512' : '[URL]',
        'MD5_512' : '344fea72eeb0870421c88140440831e9',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'
        },
        'n_channels_im' : 3,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 810, 'val' : 90, 'test' : 379},
        'license' : 'CC BY-NC 4.0',
        'reference': '10.1109/ISBI.2018.8363547, [URL]'
    }, 

    'polypgen': {
        'python_class' : "PolypGenMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : 'ff777edab5fccb224d44426364eb384c',
        'url_256' : '[URL]',
        'MD5_256' : '325423a8bfaa582ffa283c95acce6992',
        'url_512' : '[URL]',
        'MD5_512' : '43a05773a4ce5cece31bb63dfa7b897d',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'
        },
        'n_channels_im' : 3,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 984, 'val' : 140, 'test' : 288,
                       'train_C1' : 176, 'val_C1' : 24, 'test_C1' : 51,
                       'train_C2' : 190, 'val_C2' : 28, 'test_C2' : 52,
                       'train_C3' : 318, 'val_C3' : 45, 'test_C3' : 93,
                       'train_C4' : 98,  'val_C4' : 15, 'test_C4' : 33,
                       'train_C5' : 144, 'val_C5' : 20, 'test_C5' : 42,
                       'train_C6' : 58,  'val_C6' : 8,  'test_C6' : 17},
        'license' : 'CC BY 4.0',
        'annotations': {"C1": "Ambroise Paré Hospital, Paris, France",
                        "C2": "Istituto Oncologico Veneto, Padova, Italy",
                        "C3": "Centro Riferimento Oncologico, IRCCS, Italy",
                        "C4": "Oslo University Hospital, Oslo, Norway",
                        "C5": "John Radcliffe Hospital, Oxford, UK",
                        "C6": "University of Alexandria, Alexandria, Egypt"},
        'reference': '10.48550/arXiv.2202.12031, 10.1016/j.media.2021.102002'
    }, 

    'promise12': {
        'python_class' : "Promise12MSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : '879618db3641a09fa6bdec6be93e13c5',
        'url_256' : '[URL]',
        'MD5_256' : 'f5dbdbfe05b80808c4d541a63f64521c',
        'url_512' : '[URL]',
        'MD5_512' : '6943b242f5b8d65600fe6486eb71b483',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'
        },
        'n_channels_im' : 1,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 1031, 'val' : 147, 'test' : 295},
        'license' : 'Does not have copyright information',
        'reference': '10.1016/j.media.2013.12.002, 10.5281/zenodo.8014040'
    }, 

    'robotool': {
        'python_class' : "RoboToolMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : '6b7517c8637b44d11d9af7accdc3d3c8',
        'url_256' : '[URL]',
        'MD5_256' : '7e644bc28160db7e4af27b164844f136',
        'url_512' : '[URL]',
        'MD5_512' : '72d142950626d71b1df5093ae56172a8',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'
        },
        'n_channels_im' : 3,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 350, 'val' : 50, 'test' : 100},
        'license' : 'CC BY 4.0.',
        'reference' : '10.1109/TMI.2021.3057884, [URL] arXiv.1902.06426'
    }, 

    'uwaterlooskincancer': {
        'python_class' : "UWSkinCancerMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : '46e18bc41733bf5750738e418b1c7cbb',
        'url_256' : '[URL]',
        'MD5_256' : '08636b16baf28a5bbbf40a664095a2b5',
        'url_512' : '[URL]',
        'MD5_512' : '1bf86aae7d81853f973b8b7ccb8a048e',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'
        },
        'n_channels_im' : 3,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 143, 'val' : 19, 'test' : 44,
                       'train_C1' : 83, 'val_C1' : 11, 'test_C1' : 25,
                       'train_C2' : 60, 'val_C2' : 8, 'test_C2' : 19,},
        'license' : 'Non-commercial use',
        'class_names' : {"C1" : "melenoma", "C2" : "notmelenoma"},
        'reference': '[URL]'
    }, 

    'ultrasoundnerve': {
        'python_class' : "UltrasoundNerveMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : '5180ca0d1f0654765d23bc30fc073e4a',
        'url_256' : '[URL]',
        'MD5_256' : '24bca8f55e20c46c4891ae0fc92421f2',
        'url_512' : '[URL]',
        'MD5_512' : '9f6e4127e75595e9a63f0469af01a605',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'
        },
        'n_channels_im' : 1,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 1651, 'val' : 223, 'test' : 449},
        'license' : 'Does not have copyright information',
        'reference': '[URL]'
    }, 

    'usforkidney': {
        'python_class' : "USforKidneyMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : 'bc361af0f49f47a4d60b0717fb206661',
        'url_256' : '[URL]',
        'MD5_256' : '4425d8628061a254ba83659c76602f20',
        'url_512' : '[URL]',
        'MD5_512' : 'aad4fd208535c90d9a8bebe75714698c',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'
        },
        'n_channels_im' : 1,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 3210, 'val' : 458, 'test' : 918},
        'license' : 'GPL 2',
        'reference': '10.1016/j.ultras.2022.106706, [URL]'
    }, 

    'yeaz': {
        'python_class' : "YeazMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : 'e114e0f00da84e5a8940bbf7e7e06e99',
        'url_256' : '[URL]',
        'MD5_256' : 'e78c3fbb905650f7c86ecd11deec5569',
        'url_512' : '[URL]',
        'MD5_512' : '7328e0783e758d5da76729d420116a25',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'foreground'            
        },
        'n_channels_im' : 1,
        'n_channels_lbl': 1,
        'n_samples' : {'train' : 360, 'val' : 96, 'test' : 251},
        'license' : 'Does not have copyright information',
        'reference': '10.1038/s41467-020-19557-4, [URL]'
    },

    'bkai-igh': {
        'python_class' : "BkaiIghMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : '3197cdeb1a4b0d2bec47d7a8ea3afc8f ',
        'url_256' : '[URL]',
        'MD5_256' : '3c27aa13dfa8e3039fcb173c0c62cef2 ',
        'url_512' : '[URL]',
        'MD5_512' : 'f0d977e33084307683475dbc160a3dc1',
        'task'    : 'multi-class',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'neoplastic polyps',
            '2' : 'non-neoplastic polyps'           
        },
        'n_channels_im' : 3,
        'n_channels_lbl': 1,
        'n_samples' : {'train' :  700, 'val' : 100, 'test' : 200},
        'license' : 'BKAI-IGH NeoPolyp-Small is a public dataset released by BKAI Research Center, Hanoi University of Science and Technology incorporation with Institute of Gastroenterology and Hepatology (IGH), Vietnam.',
        'reference': '10.1007/978-3-030-90436-4_2, 10.1109/ACCESS.2022.3195241, arXiv:2307.06420, 10.1109/ACCESS.2022.3168693'
    },

    'm2caiseg': {
        'python_class' : "M2caiSegMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : '1e56c9f134dd97612d5cb0d3cbc8dc6b',
        'url_256' : '[URL]',
        'MD5_256' : 'e82a9d636914beea1516953aa1afbb23',
        'url_512' : '[URL]',
        'MD5_512' : '3723c8d5acb9b046c70752847f3ed426',
        'task'    : 'multi-class',
        'pixel_labels'   : {
            '0' : 'unknown',
            '1' : 'grasper',
            '2' : 'bipolar',
            '3' : 'hook',
            '4' : 'scissors',
            '5' : 'clipper',
            '6' : 'irrigator',
            '7' : 'specimen-bag',
            '8' : 'trocars',
            '9' : 'clip',
            '10': 'liver',
            '11': 'gall-bladder',
            '12': 'fat',
            '13': 'organ',
            '14': 'artery',
            '15': 'intestine',
            '16': 'bile',
            '17': 'blood',
            '18': 'black'         
        },
        'n_channels_im' : 3,
        'n_channels_lbl': 1,
        'n_samples' : {'train' :  245, 'val' : 307, 'test' : 62},
        'license' : 'CC BY-NC 4.0',
        'reference': '[URL] [URL]'
    },

    'abdomenus': {
        'python_class' : "AbdomenUSMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : '86b0bbd84f7df3726c29e6b605c83a72 ',
        'url_256' : '[URL]',
        'MD5_256' : 'c2eb21d1a93d857c472b01a72a210030 ',
        'url_512' : '[URL]',
        'MD5_512' : '6e46ed122fea25c7370d86073ea38188 ',
        'task'    : 'multi-class',
        'pixel_labels'   : {
            '0' : 'background',
            '1' : 'liver',
            '2' : 'kidney',
            '3' : 'pancreas',
            '4' : 'vessels',
            '5': 'adrenals',
            '6': 'gallbladder',
            '7': 'bones',
            '8' : 'spleen'
        },
        'n_channels_im' : 1,
        'n_channels_lbl': 1,
        'n_samples' : {'train' :  569, 'val' : 64, 'test' : 293},
        'license' : 'This dataset has been publicly released after winning a Kaggle Open Data Research Grant.',
        'reference': '10.1007/s11548-019-02046-5, [URL]'
    },

    'fhpsaop': {
        'python_class' : "FHPsAOPMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : '650747dbd726fb5560ca767af979537c',
        'url_256' : '[URL]',
        'MD5_256' : 'db6f6aa47f03dd67d2f1d23d2ce7081e',
        'url_512' : '[URL]',
        'MD5_512' : '337d11e705ae507d94fbc8f06bc15ae5',
        'task'    : 'multi-class',
        'pixel_labels'   : {
            '0': 'background',
            '1': 'pubic symphysis',
            '2': 'fetal head'
        },
        'n_channels_im' : 1,
        'n_channels_lbl': 1,
        'n_samples' : {'train' :  2800, 'val' : 400, 'test' : 800},
        'license' : 'CC BY-NC 4.0',
        'reference': '10.1016/j.dib.2022.107904,  10.5281/zenodo.7851338'
    },

    'mosmedplus': {
        'python_class' : "MosMedPlusMSBench",
        'description' : "",
        'url_128' : '[URL]',
        'MD5_128' : '537e46abb7623a1c988ef49c32f66d7d',
        'url_256' : '[URL]',
        'MD5_256' : '9a5cc1ee6c714cd64783814df5d6ec63',
        'url_512' : '[URL]',
        'MD5_512' : '15fa5aa86962803e93e6d683dcf329d5',
        'task'    : 'binary',
        'pixel_labels'   : {
            '0': 'background',
            '1': 'foreground'
        },
        'n_channels_im' : 3,
        'n_channels_lbl': 1,
        'n_samples' : {'train' :  1910, 'val' : 272, 'test' : 547},
        'license' : 'CC BY-NC-ND 3.0',
        'reference': '10.48550/arXiv.2005.06465, [URL] [URL]'
    },

}
class MedSegBench(Dataset):

    flag = ...

    def __init__(self,
                 split,
                 transform = None,
                 target_transform = None,
                 download = False,
                 root = None,
                 size=512,
                 category = None,
                 mmap_mode=None,
                ):
        
        """
        Args:

            split (string): 'train', 'val' or 'test', required
            transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Default: None.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it. Default: None.
            download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. Default: False.
            size (int, optional): The size of the returned images. Default: 512.
            category (str, optional): Sub-categories of some datasets. Default: None
            mmap_mode (str, optional): If not None, read image arrays from the disk directly. This is useful to set `mmap_mode='r'` to save memory usage when the dataset is large (e.g., PathMNIST-224). Default: None.
            root (string, optional): Root directory of dataset. Default: `~/.medsegbench`.
        """
        if size =="224":
            size=256
        self.size = size
        self.category = category
        self.size_flag = f"_{size}"
        self.info = INFO[self.flag]

        if root is not None and os.path.exists(root):
            self.root = root
        else:
            raise RuntimeError(
                "Failed to setup the default `root` directory. "
                + "Please specify and create the `root` directory manually."
            )

        if download and not os.path.exists(os.path.join(self.root, f"{self.flag}{self.size_flag}.npz")):
            self.download()
        
        if not os.path.exists(
            os.path.join(self.root, f"{self.flag}{self.size_flag}.npz")
        ):
            raise RuntimeError(
                f"Dataset not found. You can set 'download=True' to download it"
            )
        
        npz_file = np.load(
            os.path.join(self.root, f"{self.flag}{self.size_flag}.npz"),
            mmap_mode=mmap_mode,
        )

        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        if self.split in ["train", "val", "test"]:
            if self.flag == 'idrib':
                self.imgs = npz_file[f"{self.split}_images"]
            else:
                self.imgs = npz_file[f"{self.split}_images" if self.category is None else f"{self.split}_images_{self.category}"]
            self.labels = npz_file[f"{self.split}_label" if self.category is None else f"{self.split}_label_{self.category}"] 
        else:
            raise ValueError

    def __len__(self):
        assert self.info["n_samples"][self.split] == self.imgs.shape[0]
        return self.imgs.shape[0]
    
    def __getitem__(self, index):
        """
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        """

        img, target = self.imgs[index], self.labels[index]
        if self.info['task'] == 'binary': target = target * 255
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            if self.info['task'] == 'binary':
                target = self.target_transform(target)
            else:
                target = from_numpy(target).unsqueeze(0)

        return img, target

    def download(self):
        try:
            from torchvision.datasets.utils import download_url

            download_url(
                url = self.info[f"url{self.size_flag}"],
                root = self.root,
                filename = f"{self.flag}{self.size_flag}.npz",
                md5 = self.info[f"MD5{self.size_flag}"],

            )
        except:
            raise RuntimeError(
                f"""
                Automatic download failed! Please download {self.flag}{self.size_flag}.npz manually.
                1. [Optional] Check your network connection: 
                2. Download the npz file from the Zenodo repository or its Zenodo data link: 
                    {self.info[f"url{self.size_flag}"]}
                3. [Optional] Verify the MD5: 
                    {self.info[f"MD5{self.size_flag}"]}
                4. Put the npz file under your MedSegBench root folder: 
                    {self.root}
                """
            )
    
    def montage(self, length=20, replace=False, save_folder=None, display_target = False, seed = None):
        from medsegbench.utils import montage2d

        if seed is not None:
            np.random.seed(seed)

        n_sel = length * length
        sel = np.random.choice(self.__len__(), size=n_sel, replace=replace)

        montage_img = montage2d(
            imgs=self.imgs if not display_target else self.labels, n_channels=self.info["n_channels_im"] if not display_target else self.info["n_channels_lbl"], sel=sel
        )

        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            montage_img.save(
                os.path.join(
                    save_folder, f"{self.flag}{self.size_flag}_{self.split}_montage.jpg"
                )
            )

        return montage_img


class FlexibleMedSegBench(MedSegBench):
    def __init__(self, flag, split, transform=None, target_transform=None, download=False, root=None, size=512, category=None, mmap_mode=None):
        self.flag = flag
        DEFAULT_ROOT=f'./data/{flag}'
        if os.path.exists(DEFAULT_ROOT):
            print(f"Using existing root directory: {DEFAULT_ROOT}")
        else:
            print(f"Creating root directory: {DEFAULT_ROOT}")
            os.makedirs(DEFAULT_ROOT, exist_ok=True)
        root = DEFAULT_ROOT
        super().__init__(split=split, transform=transform, target_transform=target_transform, download=download, root=root, size=size, category=category, mmap_mode=mmap_mode)

    def __getitem__(self, index):


        img, target = self.imgs[index], self.labels[index]
        if self.info['task'] == 'binary': target = target * 255
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        if len(target.shape) == 2:
            target = np.expand_dims(target, axis=-1)
        augmented = self.transform(image=img, mask=target)
        image = augmented['image']
        label = augmented['mask']

        image = image.astype('float32')
        image = image.transpose(2, 0, 1) / 255

        label = label.astype('float32') 
        label = label.transpose(2, 0, 1) / 255

        label[label > 0] = 1

        sample = {"image": image, "label": label, "case": index}
        return sample




def get_MedSegBench_dataset(flag,transform,split, download=True, size=256):
    return FlexibleMedSegBench(flag=flag, split=split, transform=transform,  download=download, size=256)