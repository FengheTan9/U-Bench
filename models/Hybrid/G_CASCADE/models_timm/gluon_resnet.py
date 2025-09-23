"""Pytorch impl of MxNet Gluon ResNet/(SE)ResNeXt variants
This file evolved from [URL] 'resnet.py' with (SE)-ResNeXt additions
and ports of Gluon variations ([URL] 
by Ross Wightman
"""

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg
from .layers import SEModule
from .registry import register_model
from .resnet import ResNet, Bottleneck, BasicBlock


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    'gluon_resnet18_v1b': _cfg(url='[URL]'),
    'gluon_resnet34_v1b': _cfg(url='[URL]'),
    'gluon_resnet50_v1b': _cfg(url='[URL]'),
    'gluon_resnet101_v1b': _cfg(url='[URL]'),
    'gluon_resnet152_v1b': _cfg(url='[URL]'),
    'gluon_resnet50_v1c': _cfg(url='[URL]',
                               first_conv='conv1.0'),
    'gluon_resnet101_v1c': _cfg(url='[URL]',
                                first_conv='conv1.0'),
    'gluon_resnet152_v1c': _cfg(url='[URL]',
                                first_conv='conv1.0'),
    'gluon_resnet50_v1d': _cfg(url='[URL]',
                               first_conv='conv1.0'),
    'gluon_resnet101_v1d': _cfg(url='[URL]',
                                first_conv='conv1.0'),
    'gluon_resnet152_v1d': _cfg(url='[URL]',
                                first_conv='conv1.0'),
    'gluon_resnet50_v1s': _cfg(url='[URL]',
                               first_conv='conv1.0'),
    'gluon_resnet101_v1s': _cfg(url='[URL]',
                                first_conv='conv1.0'),
    'gluon_resnet152_v1s': _cfg(url='[URL]',
                                first_conv='conv1.0'),
    'gluon_resnext50_32x4d': _cfg(url='[URL]'),
    'gluon_resnext101_32x4d': _cfg(url='[URL]'),
    'gluon_resnext101_64x4d': _cfg(url='[URL]'),
    'gluon_seresnext50_32x4d': _cfg(url='[URL]'),
    'gluon_seresnext101_32x4d': _cfg(url='[URL]'),
    'gluon_seresnext101_64x4d': _cfg(url='[URL]'),
    'gluon_senet154': _cfg(url='[URL]',
                           first_conv='conv1.0'),
}


def _create_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ResNet, variant, pretrained, **kwargs)


@register_model
def gluon_resnet18_v1b(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)
    return _create_resnet('gluon_resnet18_v1b', pretrained, **model_args)


@register_model
def gluon_resnet34_v1b(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)
    return _create_resnet('gluon_resnet34_v1b', pretrained, **model_args)


@register_model
def gluon_resnet50_v1b(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_resnet('gluon_resnet50_v1b', pretrained, **model_args)


@register_model
def gluon_resnet101_v1b(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)
    return _create_resnet('gluon_resnet101_v1b', pretrained, **model_args)


@register_model
def gluon_resnet152_v1b(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], **kwargs)
    return _create_resnet('gluon_resnet152_v1b', pretrained, **model_args)


@register_model
def gluon_resnet50_v1c(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', **kwargs)
    return _create_resnet('gluon_resnet50_v1c', pretrained, **model_args)


@register_model
def gluon_resnet101_v1c(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', **kwargs)
    return _create_resnet('gluon_resnet101_v1c', pretrained, **model_args)


@register_model
def gluon_resnet152_v1c(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep', **kwargs)
    return _create_resnet('gluon_resnet152_v1c', pretrained, **model_args)


@register_model
def gluon_resnet50_v1d(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True, **kwargs)
    return _create_resnet('gluon_resnet50_v1d', pretrained, **model_args)


@register_model
def gluon_resnet101_v1d(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', avg_down=True, **kwargs)
    return _create_resnet('gluon_resnet101_v1d', pretrained, **model_args)


@register_model
def gluon_resnet152_v1d(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep', avg_down=True, **kwargs)
    return _create_resnet('gluon_resnet152_v1d', pretrained, **model_args)


@register_model
def gluon_resnet50_v1s(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=64, stem_type='deep', **kwargs)
    return _create_resnet('gluon_resnet50_v1s', pretrained, **model_args)



@register_model
def gluon_resnet101_v1s(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], stem_width=64, stem_type='deep', **kwargs)
    return _create_resnet('gluon_resnet101_v1s', pretrained, **model_args)


@register_model
def gluon_resnet152_v1s(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], stem_width=64, stem_type='deep', **kwargs)
    return _create_resnet('gluon_resnet152_v1s', pretrained, **model_args)



@register_model
def gluon_resnext50_32x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt50-32x4d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4, **kwargs)
    return _create_resnet('gluon_resnext50_32x4d', pretrained, **model_args)


@register_model
def gluon_resnext101_32x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=4, **kwargs)
    return _create_resnet('gluon_resnext101_32x4d', pretrained, **model_args)


@register_model
def gluon_resnext101_64x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=64, base_width=4, **kwargs)
    return _create_resnet('gluon_resnext101_64x4d', pretrained, **model_args)


@register_model
def gluon_seresnext50_32x4d(pretrained=False, **kwargs):
    """Constructs a SEResNeXt50-32x4d model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4,
        block_args=dict(attn_layer=SEModule), **kwargs)
    return _create_resnet('gluon_seresnext50_32x4d', pretrained, **model_args)


@register_model
def gluon_seresnext101_32x4d(pretrained=False, **kwargs):
    """Constructs a SEResNeXt-101-32x4d model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=4,
        block_args=dict(attn_layer=SEModule), **kwargs)
    return _create_resnet('gluon_seresnext101_32x4d', pretrained, **model_args)


@register_model
def gluon_seresnext101_64x4d(pretrained=False, **kwargs):
    """Constructs a SEResNeXt-101-64x4d model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=64, base_width=4,
        block_args=dict(attn_layer=SEModule), **kwargs)
    return _create_resnet('gluon_seresnext101_64x4d', pretrained, **model_args)


@register_model
def gluon_senet154(pretrained=False, **kwargs):
    """Constructs an SENet-154 model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], cardinality=64, base_width=4, stem_type='deep',
        down_kernel_size=3, block_reduce_first=2, block_args=dict(attn_layer=SEModule), **kwargs)
    return _create_resnet('gluon_senet154', pretrained, **model_args)