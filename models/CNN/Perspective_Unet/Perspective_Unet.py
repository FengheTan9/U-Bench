import math
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange, repeat
from functools import partial


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), stride=stride, bias=bias)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    some = True
    q, r = torch.linalg.qr(unstructured_block.cpu(), 'reduced' if some else 'complete')
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix


def generalized_kernel(data, *, projection_matrix, kernel_fn=nn.ReLU(), kernel_epsilon=0.001, normalize_data=True):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)


def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=False, eps=1e-4, device=None):
    b, h, *_ = data.shape

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', data, projection)
    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0)
    diag_data = diag_data.unsqueeze(dim=-1)

    data_dash = ratio * (torch.exp(data_dash - diag_data) + eps)

    return data_dash.type_as(data)


# non-causal linear attention
def linear_attention(q, k, v):
    k_cumsum = k.sum(dim=-2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out


class ENLA(nn.Module):
    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0, generalized_attention=False, kernel_fn=nn.ReLU(),
                 no_projection=False, attn_drop=0.):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows=self.nb_features,
                                         nb_columns=dim_heads, scaling=ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection
        self.attn_drop = nn.Dropout(attn_drop)

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        # q[b,h,n,d],b is batch ,h is multi head, n is number of batch, d is feature
        device = q.device

        if self.no_projection:
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-2)

        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn=self.kernel_fn,
                                    projection_matrix=self.projection_matrix, device=device)
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix, device=device)
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)

        attn_fn = linear_attention
        out = attn_fn(q, k, v)
        out = self.attn_drop(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Element-wise addition
        out = F.relu(out)
        return out


class DilatedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2):
        super(DilatedResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Element-wise addition
        out = F.relu(out)
        return out


class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionBlock, self).__init__()
        # Assuming the input from each block is concatenated along the channel dimension
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)  # Concatenate along channel dimension
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BiPathResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dilation=2, use_dilate_conv=True):
        super(BiPathResBlock, self).__init__()
        # Define two ResBlocks and two DilatedResBlocks in sequence for each path
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            ResBlock(mid_channels, mid_channels)
        )
        self.dilated_resblock = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            DilatedResBlock(mid_channels, mid_channels, dilation=dilation)
        )
        # Define the Fusion Block
        self.fusionblock = FusionBlock(2 * mid_channels, out_channels)
        self.use_dilate_conv = use_dilate_conv

    def forward(self, x):
        res_out = self.resblock(x)
        dilated_res_out = self.dilated_resblock(x)
        if self.use_dilate_conv:
            x = self.fusionblock(res_out, dilated_res_out)
        else:
            x = self.fusionblock(res_out, res_out)
        return x


class CNNEncoder(nn.Module):
    def __init__(self, use_dilate_conv=True):
        super(CNNEncoder, self).__init__()

        # Define channel transitions from the input to the deepest layer
        channels = [3, 64, 128, 256, 512, 1024]
        self.layers = nn.ModuleList()

        for idx in range(1, len(channels)):
            self.layers.append(BiPathResBlock(channels[idx - 1], channels[idx], channels[idx], use_dilate_conv=use_dilate_conv))
            if idx != len(channels) - 1:
                self.layers.append(nn.MaxPool2d(2))

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, BiPathResBlock):  # Conditionally append feature maps following DoubleResBlock layers
                features.append(x)
        # # Include the final feature map post application of MaxPool2d layer for completeness of the hierarchical representations
        # features.append(x)

        return features


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicBlock(nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, bias=True, bn=False, act=None):
        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class ENLTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        # self.mlp_ratio = mlp_ratio
        self.qk_scale = qk_scale
        self.conv_match1 = BasicBlock(default_conv, dim, dim, kernel_size, bias=qkv_bias, bn=False, act=None)
        self.conv_match2 = BasicBlock(default_conv, dim, dim, kernel_size, bias=qkv_bias, bn=False, act=None)
        self.conv_assembly = BasicBlock(default_conv, dim, dim, kernel_size, bias=qkv_bias, bn=False, act=None)

        self.norm1 = norm_layer(dim)
        self.attn = ENLA(dim_heads=dim, nb_features=dim, attn_drop=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        assert H == x.shape[-2] and W == x.shape[-1], "input feature has wrong size"
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
        shortcut = x  # skip connection

        # Layer Norm
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()

        # ENLA
        x_embed_1 = self.conv_match1(x)
        x_embed_2 = self.conv_match2(x)
        x_assembly = self.conv_assembly(x)  # [B,C,H,W]
        if self.qk_scale is not None:
            x_embed_1 = F.normalize(x_embed_1, p=2, dim=1, eps=5e-5) * self.qk_scale
            x_embed_2 = F.normalize(x_embed_2, p=2, dim=1, eps=5e-5) * self.qk_scale
        else:
            x_embed_1 = F.normalize(x_embed_1, p=2, dim=1, eps=5e-5)
            x_embed_2 = F.normalize(x_embed_2, p=2, dim=1, eps=5e-5)
        B, C, H, W = x_embed_1.shape
        x_embed_1 = x_embed_1.permute(0, 2, 3, 1).view(B, 1, H * W, C)
        x_embed_2 = x_embed_2.permute(0, 2, 3, 1).view(B, 1, H * W, C)
        x_assembly = x_assembly.permute(0, 2, 3, 1).view(B, 1, H * W, -1)

        x = self.attn(x_embed_1, x_embed_2, x_assembly).squeeze(1)  # (B, H*W, C)
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x


class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        self.block_layer = [2, 2, 2]
        self.size = [64, 32, 16]
        self.channels = [256, 512, 1024]
        self.num_heads = [6, 6, 6]

        self.stages = nn.ModuleList([
            self._make_stage(num_blocks, size, dim, num_heads) for num_blocks, size, dim, num_heads
            in zip(self.block_layer, self.size, self.channels, self.num_heads)
        ])

        self.downsample = nn.ModuleList([
            ConvBNReLU(self.channels[i], self.channels[i] * 2, 2, 2, padding=0) for i in
            range(len(self.block_layer) - 1)
        ])

        self.channel_adjuster = nn.ModuleList([
            nn.Conv2d(self.channels[i] * 4, self.channels[i] * 2, 1, 1) for i in range(len(self.block_layer) - 2)
        ])
        self.channel_adjuster.append(nn.Conv2d(self.channels[-1] * 2, self.channels[-1], 1, 1))

    @staticmethod
    def _make_stage(num_blocks, size, dim, num_heads):
        return nn.Sequential(*[
            ENLTransformerBlock(
                input_resolution=(size, size),
                dim=dim,
                num_heads=num_heads,
                drop_path=0.1,
                drop=0.1,
                attn_drop=0.1
            ) for _ in range(num_blocks)
        ])

    def forward(self, x):
        _, _, feature0, feature1, feature2 = x
        features = [feature0, feature1, feature2]

        transformed_features = []

        for idx, (feature, stage) in enumerate(zip(features, self.stages)):
            if idx == 0:
                feature_trans = stage(feature)
            else:
                feature_down = self.downsample[idx - 1](transformed_features[-1])
                feature_in = torch.cat((feature, feature_down), dim=1)
                feature_in = self.channel_adjuster[idx - 1](feature_in)
                feature_trans = stage(feature_in)

            transformed_features.append(feature_trans)
        return transformed_features


class SpatialTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class SpatialTransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([SpatialTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                                     for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        src = self.norm(src)
        return src


class SpatialCrossScaleIntegrator(nn.Module):
    def __init__(self, channel_sizes, target_dim=256, nhead=8, depth=6):
        super().__init__()
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(in_channels, target_dim, 1) for in_channels in channel_sizes
        ])
        # Positional encodings are not used for simplicity.
        # Implement positional encoding if necessary.
        self.transformer = SpatialTransformerEncoder(d_model=target_dim, nhead=nhead, num_layers=depth)
        self.reproj_layers = nn.ModuleList([
            nn.Conv2d(target_dim, out_channels, 1) for out_channels in channel_sizes
        ])
        self.h_w = [(64, 64), (32, 32), (16, 16)]  # Hardcoded sizes for outputs
        self.target_dim = target_dim

    def forward(self, inputs):
        # Project and flatten
        flatten_features = []
        for i, x in enumerate(inputs):
            x_proj = self.proj_layers[i](x)
            flatten_features.append(rearrange(x_proj, 'b c h w -> b (h w) c'))

        # Combine & Transform
        combined = torch.cat(flatten_features, dim=1)
        split_sizes = [h * w for h, w in self.h_w]
        transformed = self.transformer(combined)

        hs_ws_splits = torch.split(transformed, split_sizes, dim=1)
        outputs = []
        for i, split in enumerate(hs_ws_splits):
            split = rearrange(split, f'b (h w) c -> b c h w', h=self.h_w[i][0], w=self.h_w[i][1])
            out = self.reproj_layers[i](split)
            outputs.append(out)
        return outputs


class Encoder(nn.Module):
    def __init__(self, use_enltb=True, use_scale_integrator=True, use_dilate_conv=True):
        super(Encoder, self).__init__()
        self.Encoder1 = CNNEncoder(use_dilate_conv=use_dilate_conv)
        self.Encoder2 = TransformerEncoder() if use_enltb else None
        self.num_module = 3
        self.fusion_list = [256, 512, 1024]
        self.use_scale_integrator = use_scale_integrator
        self.scale_integrator = SpatialCrossScaleIntegrator(self.fusion_list) if use_scale_integrator else None

        self.fuser = nn.ModuleList()
        for i in range(self.num_module):
            self.fuser.append(
                nn.Conv2d(self.fusion_list[i] * 2, self.fusion_list[i], 1, 1)
            )

    def forward(self, x):
        skips = []
        features = self.Encoder1(x)
        feature_trans = self.Encoder2(features) if self.Encoder2 is not None else features[-3:]

        if self.scale_integrator is not None:
            feature_trans = self.scale_integrator(feature_trans)
        skips.extend(features[:2])
        for i in range(self.num_module):
            skip = self.fuser[i](torch.cat((feature_trans[i], features[i + 2]), dim=1))
            skips.append(skip)
        return skips


class Decoder(nn.Module):
    class DecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = ConvBNReLU(in_channels, out_channels, 3, stride=1, padding=1)
            self.conv2 = ConvBNReLU(out_channels, out_channels, 3, stride=1, padding=1)
            self.upscale = nn.UpsamplingBilinear2d(scale_factor=2)
            self.in_channels = in_channels

        def forward(self, x, skip=None):
            x = self.upscale(x)
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
            x = self.conv1(x)
            x = self.conv2(x)
            return x

    class LastDecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = ConvBNReLU(in_channels, out_channels, 3, stride=1, padding=1)
            self.conv2 = ConvBNReLU(out_channels, out_channels, 3, stride=1, padding=1)
            self.in_channels = in_channels

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            return x

    def __init__(self):
        super().__init__()
        self.encoder_channels = [512, 256, 128, 64]
        self.first_block = self.LastDecoderBlock(1024, self.encoder_channels[0])
        self.blocks = nn.ModuleList([
            self.DecoderBlock(2 * in_ch, out_ch)
            for in_ch, out_ch in zip(self.encoder_channels[:-1], self.encoder_channels[1:])
        ])
        self.last_block = self.DecoderBlock(self.encoder_channels[-1], self.encoder_channels[-1])

    def forward(self, encoder_skips):
        x = encoder_skips[-1]  # Start from the deepest feature map
        x = self.first_block(x)
        for skip, block in zip(reversed(encoder_skips[:-1]), self.blocks):
            x = block(x, skip)
        x = self.last_block(x)

        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class PUnet(nn.Module):
    def __init__(self, num_classes, use_enltb=True, use_scale_integrator=True, use_dilate_conv=True, input_channel=3):
        super().__init__()
        self.encoder = Encoder(use_enltb=use_enltb, use_scale_integrator=use_scale_integrator, use_dilate_conv=use_dilate_conv)
        self.decoder = Decoder()
        self.segmentation_head = SegmentationHead(in_channels=64, out_channels=num_classes, kernel_size=3,)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        encoder_skips = self.encoder(x)
        out = self.decoder(encoder_skips)
        logits = self.segmentation_head(out)
        return logits

def perspective_unet(num_classes, input_channel=3):
    model = PUnet(input_channel=input_channel, num_classes=num_classes)
    return model
