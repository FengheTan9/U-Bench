import torch
import torch.utils.model_zoo as model_zoo

# 使用相对导入，一个点表示当前包
from . import backbone
model_urls = {
    'resnet18': '[URL]',
    'resnet50': '[URL]',
    'res2net50_v1b_26w_4s': '[URL]',
    'res2net101_v1b_26w_4s': '[URL]',
    'resnest50': '[URL]'
}

def IS2D_model(args) :
    from mfmsnet import MFMSNet
    return MFMSNet(args.num_classes,
                   args.scale_branches,
                   args.frequency_branches,
                   args.frequency_selection,
                   args.block_repetition,
                   args.min_channel,
                   args.min_resolution,
                   args.cnn_backbone)

def load_cnn_backbone_model(backbone_name, pretrained=False, **kwargs):
    if backbone_name=='resnet18':
        from .backbone.resnet import ResNet
        model = ResNet(backbone.resnet.BasicBlock, [2, 2, 2, 2], **kwargs)
    elif backbone_name=='resnet50':
        from .backbone.resnet import ResNet
        model = ResNet(backbone.resnet.Bottleneck, [3, 4, 6, 3], **kwargs)
    elif backbone_name=='res2net50_v1b_26w_4s':
        from .backbone.res2net import Res2Net
        model = Res2Net(backbone.res2net.Bottle2Neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    elif backbone_name=='res2net101_v1b_26w_4s':
        from .backbone.res2net import Res2Net
        model = Res2Net(backbone.res2net.Bottle2Neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    elif backbone_name=='resnest50':
        from .backbone.resnest import ResNeSt
        model = ResNeSt(backbone.resnest.Bottleneck, [3, 4, 6, 3],
                        radix=2, groups=1, bottleneck_width=64,
                        deep_stem=True, stem_width=32, avg_down=True,
                        avd=True, avd_first=False, **kwargs)
    else:
        import sys
        print("Invalid backbone")
        sys.exit()

    if pretrained:
        if backbone_name == 'resnest50':
            _url_format = '[URL]'

            _model_sha256 = {name: checksum for checksum, name in [
                ('528c19ca', 'resnest50'),
                ('22405ba7', 'resnest101'),
                ('75117900', 'resnest200'),
                ('0cc87c48', 'resnest269'),
            ]}

            def short_hash(name):
                if name not in _model_sha256:
                    raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
                return _model_sha256[name][:8]

            resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
                                  name in _model_sha256.keys()
                                  }

            model.load_state_dict(torch.hub.load_state_dict_from_url(
                resnest_model_urls['resnest50'], progress=True, check_hash=True))
        else:
            model.load_state_dict(model_zoo.load_url(model_urls[backbone_name]))

        print("Complete loading your pretrained backbone {}".format(backbone_name))
    return model

def model_to_device(args, model):
    model = model.to(args.device)

    return model