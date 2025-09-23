import inspect
import argparse


from .CNN.MEGANet_ResNet.EGANet import eganet as MEGANet
from .CNN.SimpleUNet.SimpleUNet import SimpleUNet # (input_channel=3, num_classes=1)
from .CNN.ULite.ULite import ULite # (input_channel=3, num_classes=1)
from .CNN.MMUNet.MMUNet import mmunet as MMUNet # (input_channel=3, num_classes=1)
from .CNN.UACANet.UACANet import UACANet # (input_channel=3, num_classes=1)
from .CNN.CSCAUNet.CSCAUNet import CSCAUNet # (input_channel=3, num_classes=1)
from .CNN.UNet_v2.UNet_v2 import UNetV2
from .CNN.RollingUnet.RollingUnet import Rolling_Unet_M as RollingUnet
from .CNN.dobuleunet.dobuleunet import build_doubleunet as DoubleUNet
from .CNN.AttU_Net.AttU_Net import AttU_Net # (input_channel=3, num_classes=1)
from .CNN.CMUNeXt.CMUNeXt import CMUNeXt # (input_channel=3, num_classes=1)
from .CNN.CMU_Net.CMU_Net import CMU_Net # (input_channel=3, num_classes=1)
from .CNN.UNeXt.UNeXt import UNeXt # (num_classes=1, input_channel=3)
from .CNN.UNet3plus.UNet3plus import UNet3plus # (input_channel=3, num_classes=1)
from .CNN.UNetplus.UNetplus import ResNet34UnetPlus # (input_channel=3, num_classes=1)
from .CNN.U_Net.U_Net import U_Net # (input_channel=3, num_classes=1)
from .CNN.Tinyunet.Tinyunet import Tinyunet # (input_channel=3, num_classes=1)
from .CNN.Egeunet.Egeunet import EGEUNet as Egeunet # (input_channel=3, num_classes=1)
from .CNN.ERDUnet.ERDUnet import ERDUnet # (input_channel=3, num_classes=1)
from .CNN.IS2D_models.mfmsnet import MFMSNet # (num_classes=1, input_channel=1, scale_branches=3, frequency_branches=16, frequency_selection='top', frequency_selection_ratio=0.5, use_deepsupervision=True)
from .CNN.TA_Net.TA_Net import TA_Net
from .CNN.DDANet.DDANet import ddanet as DDANet
from .CNN.PraNet.PraNet import PraNet # (input_channel=3, num_classes=1, pretrained=True, freeze_encoder=False, deep_supervision=False) # (ch=64, num_classes=1,, pretrained=True, freeze_encoder=False, deep_supervision=False)
from .CNN.TernausNet.TernausNet import ternausnet
from .CNN.R2U_Net.R2U_Net import r2unet as R2U_Net
from .CNN.CE_Net.CE_Net import ce_net as CE_Net
from .CNN.MultiResUnet.MultiResUnet import multiresunet as MultiResUNet
from .CNN.ResUnetPlusPlus.ResUnetPlusPlus import resunetplusplus as ResUNetPlusPlus
from .CNN.MBSNet.MBSNet import mbsnet as MBSNet
from .CNN.CA_Net.CA_Net import ca_net as CA_Net
from .CNN.KiU_Net.KiU_Net import kiu_net
from .CNN.LFU_Net.LFU_Net import lfu_net as LFU_Net
from .CNN.DC_Unet.DC_Unet import dc_unet as DC_UNet
from .CNN.ColonSegNet.ColonSegNet import colonsegnet as ColonSegNet
from .CNN.MALUNet.MALUNet import malunet as MALUNet
from .CNN.DCSAU_Net.DCSAU_Net import dcsau_net as DCSAU_Net
from .CNN.FAT_Net.FAT_Net import fat_net as FAT_Net
from .CNN.CFPNet_M.CFPNet_M import cfpnet_m as CFPNet_M
from .CNN.CaraNet.CaraNet import caranet as CaraNet
from .CNN.GH_UNet.GH_UNet import gh_unet as GH_UNet
from .CNN.MSRFNet.MSRFNet import msrfnet as MSRFNet
from .CNN.LV_UNet.LV_UNet import lv_unet as LV_UNet
from .CNN.Perspective_Unet.Perspective_Unet import perspective_unet as Perspective_Unet
from .CNN.ESKNet.ESKNet import esknet as ESKNet
from .CNN.CPCANet.CPCANet import cpcanet as CPCANet
from .CNN.UTANet.UTANet import utanet as UTANet
from .CNN.DDS_UNet.DDS_UNet import dds_unet as DDS_UNet
from .CNN.MCA_UNet.MCA_UNet import mca_unet as MCA_UNet
from .CNN.MDSA_UNet.MDSA_UNet import mdsa_unet as MDSA_UNet
from .CNN.U_KAN.U_KAN import u_kan as U_KAN
from .CNN.ResU_KAN.ResU_KAN import resu_kan as ResU_KAN
from .CNN.RAT_Net.RAT_Net import rat_net as RAT_Net

from .Hybrid.AURA_Net.AURA_Net import aura_net as AURA_Net
from .Hybrid.BEFUnet.BEFUnet import befunet as BEFUnet
from .Hybrid.CASCADE.CASCADE import cascade as CASCADE
from .Hybrid.G_CASCADE.G_CASCADE import g_cascade as G_CASCADE
from .Hybrid.ConvFormer.ConvFormer import convformer as ConvFormer
from .Hybrid.DA_TransUNet.DA_TransUNet import da_transformer as DA_TransUNet
from .Hybrid.DAEFormer.DAEFormer import daeformer as DAEFormer
from .Hybrid.DS_TransUNet.DS_TransUNet import ds_transunet as DS_TransUNet
from .Hybrid.FCBFormer.FCBFormer import fcbformer as FCBFormer
from .Hybrid.HiFormer.HiFormer import hiformer as HiFormer
from .Hybrid.LeViT_UNet.LeViT_UNet import levit_unet as LeViT_UNet
from .Hybrid.MERIT.MERIT import merit as MERIT
from .Hybrid.MT_UNet.MT_UNet import mt_unet as MT_UNet
from .Hybrid.TransAttUnet.TransAttUnet import trans_attention_unet as TransAttUnet
from .Hybrid.TransFuse.TransFuse import transfuse as TransFuse
from .Hybrid.TransNorm.TransNorm import transnorm as TransNorm
from .Hybrid.TransResUNet.TransResUNet import trans_res_unet as TransResUNet
from .Hybrid.UTNet.UTNet import utnet as UTNet
from .Hybrid.UCTransNet.UCTransNet import UCTransNet # (input_channel=3, n_classes=1, img_size=256)
from .Hybrid.EMCAD.networks import EMCADNet as EMCAD
from .Hybrid.CSWin_UNet.CSWin_UNet import cswin_unet as CSWin_UNet
from .Hybrid.D_TrAttUnet.D_TrAttUnet import d_trattunet as D_TrAttUnet
from .Hybrid.EViT_UNet.EViT_UNet import evit_unet as EViT_UNet
from .Hybrid.MedFormer.MedFormer import medformer as MedFormer
from .Hybrid.MSLAU_Net.MSLAU_Net import mslau_net as MSLAU_Net
from .Hybrid.MissFormer.MissFormer import Missformer as MissFormer
from .Hybrid.TransUnet.TransUnet import transunet as TransUnet
from .Hybrid.MobileUViT.MobileUViT import mobileuvit_l as MobileUViT
from .Hybrid.LGMSNet.LGMSNet import lgmsnet as LGMSNet
from .Hybrid.SwinUNETR.SwinUNETR import swinunetr as SwinUNETR
from .Hybrid.UNETR.UNETR import unetr as UNETR
from .Hybrid.CFFormer.CFFormer import cfformer as CFFormer
from .Hybrid.CENet.CENet import cenet as CENet
from .Hybrid.H2Former.H2Former import h2former as H2Former
from .Hybrid.ScribFormer.ScribFormer import scribformer as ScribFormer
# from .Hybrid.BRAUnet_plus_plus.bra_unet import braunet_plus_plus  #  bug  

from .Transformer.BATFormer.BATFormer import batformer as BATFormer
from .Transformer.Polyp_PVT.Polyp_PVT import polyp_pvt as Polyp_PVT
from .Transformer.SCUNet_plus_plus.SCUNet_plus_plus import scunet_plus_plus as SCUNet_plus_plus
from .Transformer.SwinUnet.SwinUnet import swinunet as SwinUnet
from .Transformer.MedT.MedT import medt as MedT


from .Mamba.AC_MambaSeg.AC_MambaSeg import ac_mambaseg as AC_MambaSeg
from .Mamba.H_vmunet.H_vmunet import h_vmunet as H_vmunet
from .Mamba.MambaUnet.MambaUnet import mambaunet as MambaUnet
from .Mamba.MUCM_Net.MUCM_Net import mucm_net as MUCM_Net
from .Mamba.Swin_umamba.Swin_umamba import swin_umamba as Swin_umamba
from .Mamba.Swin_umambaD.Swin_umambaD import swin_umambad as Swin_umambaD
from .Mamba.UltraLight_VM_UNet.UltraLight_VM_UNet import ultralight_vm_unet as UltraLight_VM_UNet
from .Mamba.VMUNet.VMUNet import vmunet as VMUNet
from .Mamba.VMUNetV2.VMUNetV2 import vmunetv2 as VMUNetV2
from .Mamba.CFM_UNet.CFM_UNet import cfm_unet as CFM_UNet
from .Mamba.MedVKAN.MedVKAN import medvkan as MedVKAN


def load_model_lazily(config):
    if config.model == "Zig_RiR":
        from .RWKV.Zig_RiR.Zig_RiR import zig_rir as Zig_RiR
        return Zig_RiR(input_channel=config.input_channel,num_classes=config.num_classes)

    elif config.model == "RWKV_UNet":
        from .RWKV.RWKV_UNet.RWKV_UNet import rwkv_unet as RWKV_UNet
        return RWKV_UNet(input_channel=config.input_channel,num_classes=config.num_classes)

    elif config.model == "U_RWKV":
        from .RWKV.U_RWKV.U_RWKV import u_rwkv as U_RWKV
        return U_RWKV(input_channel=config.input_channel,num_classes=config.num_classes)

    else:
        raise ValueError(f"Unsupported model name: {config.model}. Supported models are: 'zig_rir', 'rwkv_unet', 'u_rwkv'.")
        return None



def load_model_id(modelname):
    """读取model_id.json"""
    import json
    import os
    model_id_path = './models/model_id.json'
    with open(model_id_path, 'r') as f:
        model_ids = json.load(f)
    for model_info in model_ids:
        if model_info['modelname'] == modelname:
            deep_supervision = model_info.get('deeps_supervision', 0)
            id = model_info.get('id', None)
            if id is None:
                print(f"Model {modelname} does not have a valid model_id.")
                return None
            return id,deep_supervision
    print(f"Model {modelname} not found in model_id.json.")
    return None



def build_model(config,**kwargs):

    model_name= config.model
    
    if model_name in ['Zig_RiR', 'RWKV_UNet', 'U_RWKV']:
        return load_model_lazily(config)
    model_id,config.do_deeps = load_model_id(model_name)
    print(f"Building model {model_name} with model_id {model_id} and do_deeps {config.do_deeps}")
    if model_id is not None:
        print(f"Using model_id {model_id} for model {model_name}")
        config.model_id = model_id
    else:
        print(f"No model_id found for model {model_name}, using default.")
        exit(0)

    

    # Get the model class from the current module's globals
    if model_name not in globals():
        raise ValueError(f"Model {model_name} not found. Available models: {list(filter(lambda x: not x.startswith('_'), globals().keys()))}")
    
    model_class = globals()[model_name]
    
    # Get the signature of the model's constructor
    sig = inspect.signature(model_class.__init__)
    
    # Filter out kwargs that are not in the constructor's signature
    model_args = {k: v for k, v in kwargs.items() if k in sig.parameters}
    print(f" kwargs: {model_args}")
    if 'self' not in model_args:
        return model_class(**kwargs)


    # Check for unexpected arguments
    unexpected_args = {k: v for k, v in kwargs.items() if k not in sig.parameters and k != 'self'}
    if unexpected_args:
        raise TypeError(f"Got unexpected keyword arguments: {list(unexpected_args.keys())}")
    print(f"Building model {model_name} with arguments: {model_args}")
    return model_class(**model_args)
