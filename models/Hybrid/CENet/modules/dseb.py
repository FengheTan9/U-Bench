import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from .multihead_diffattn import MultiheadDiffAttn





# class DoGEdge(nn.Module):
#     def __init__(self,dim: int, scale_factors: list) -> None:
#         super().__init__()
#         self.scale_factors = scale_factors
#         self.w = nn.Parameter(torch.ones(dim, 1, 1)*0.5)
#     def forward(self, x):
#         _, C, H, W = x.shape
#         x_1 = F.interpolate(x, scale_factor = self.scale_factors[0], mode = "bilinear")
#         x_2 = F.interpolate(x, scale_factor = self.scale_factors[1], mode = "bilinear")
#         x_1 = F.interpolate(x_1, size=(H, W), mode = 'bilinear')
#         x_2 = F.interpolate(x_2, size=(H, W), mode = 'bilinear')  
#         x_hf = torch.abs(x_1-x_2)
#         x = x + self.w*x_hf
#         return x
    
class FEA(nn.Module):
    def __init__(self, dim: int, scale_factors: list, label="", writer=None) -> None:
        super().__init__()
        self.writer = writer
        self.label = label
        self.scale_factors = scale_factors
        self.n = n = len(scale_factors)
        self.m = m = n*(n-1)//2
        # self.ew = nn.Parameter(torch.randn(m, 1, 1, 1, 1))
        self.w = nn.Parameter(torch.randn(1, dim, 1, 1)+0.5)
        # Get the indices of the upper triangle (j > i)
        self.indices = torch.triu_indices(row=self.n, col=self.n, offset=1)
        self.global_step = 0
    
    def compute_weighted_edges(self, edges_stack):
        # edges_stack: (m, B, C, H, W)

        # Compute the pairwise absolute differences. diff_matrix: (m, m, B, C, H, W)
        diff_matrix = torch.abs(edges_stack[:, None, ...] - edges_stack[None, :, ...])

        w_edge = 0
        for idx, (i, j) in enumerate(zip(self.indices[0], self.indices[1])):
            # w_edge += F.sigmoid(1e1*self.ew[idx]) * diff_matrix[i, j, ...]
            w_edge += 1./self.m * diff_matrix[i, j, ...]
        return w_edge
    
    def write_info(self):
        # write the weights of the edges to tensorboard
        for idx, (i, j) in enumerate(zip(self.indices[0], self.indices[1])):
            title = f"EdgeWeights/L:{self.label}_({self.scale_factors[i]}-{self.scale_factors[j]})"
            self.writer.add_scalar(title, F.sigmoid(1e1*self.ew[i]).mean().item(), self.global_step)
            # title = f"EdgeWeights/L:{self.label},Std({self.scale_factors[i]}-{self.scale_factors[j]})"
            # self.writer.add_scalar(title, F.sigmoid(1e1*self.ew[i]).std().item(), self.global_step)
            # title = f"EdgeWeights/L:{self.label},Hist({self.scale_factors[j]}-{self.scale_factors[i]})"
            # self.writer.add_histogram(title, F.sigmoid(1e1*self.ew[i]).flatten(), self.global_step)
        self.global_step += 1
    
    def forward(self, x):
        _, C, H, W = x.shape
        edges = []
        for scale in self.scale_factors:
            x_1 = F.interpolate(x, scale_factor=scale, mode="bilinear")
            x_1 = F.interpolate(x_1, size=(H, W), mode='bilinear')
            edges.append(torch.abs(x-x_1))
        edge = self.compute_weighted_edges(torch.stack(edges))
        
        # write the weights of the edges to tensorboard if it's train mode and writer is not None
        # if self.writer is not None and self.train:
        #     self.write_info()
        
        return x + self.w*edge


class DSEBlock(nn.Module):
    '''
    if use_command is 'no', the block will not be used
    if use_command is 'dat', the block will only use DiffAttn
    if use_command is 'fea', the block will only use Feature Edge Amplification
    if use_command is 'dat-fea', the block will use both DiffAttn and Feature Edge Amplification
    if use_command is 'dog', the block will use difference of Gaussian (with learnable sigma) for feature sharpening
    if ues_command is 'dat-dog', the block will use both DiffAttn and difference of Gaussian
    if use_command includes 'seq', the block will use sequential attention (last is DiffAttn)
    '''
    def __init__(self, dim, scale_factors, num_heads, input_size, mode='add', use_command='dat-fea', depth=1, label="", writer=None):
        super().__init__()
        self.use_command = use_command
        self.not_use_this = 'no' in self.use_command.lower()
        if self.not_use_this:
            return
        self.input_size = input_size
        self.mode = mode.lower()
        _dim = dim*2 if self.mode=='cat' else dim

        self.use_diffattn = 'dat' in self.use_command.lower()
        self.use_fea = 'fea' in self.use_command.lower()
        self.use_dog = 'dog' in self.use_command.lower()
        self.do_seq = 'seq' in self.use_command.lower()

        if self.use_fea:
            self.boundary = FEA(dim=_dim, scale_factors=scale_factors, label=label, writer=writer)
        if self.use_diffattn:
            self.diffattn = MultiheadDiffAttn(embed_dim=_dim, depth=depth, num_heads=num_heads)
        if self.use_dog:
            self.sigma_raw_1 = nn.Parameter(torch.randn(1, _dim, 1,1)-.4, requires_grad=True)
            self.sigma_raw_2 = nn.Parameter(torch.randn(1, _dim, 1,1)-.0, requires_grad=True)
        self.mixer = nn.Conv2d(in_channels=_dim, out_channels=dim, kernel_size=1, stride=1, bias=False)


    def apply_diffattn(self, x):
        y_token = x.view(x.shape[0], -1, x.shape[1])
        diff = self.diffattn(y_token) #* y_token
        diff = diff.view(diff.shape[0], diff.shape[2], diff.shape[1]//self.input_size, diff.shape[1]//self.input_size)
        return diff * x
    
    def get_sigma(self, sigma):
        # Apply sigmoid to keep values in (0, 1), then scale to (0, 2)
        sigma = 2 * torch.sigmoid(sigma)  
        return sigma  # Now sigma is in range [0, 2]
    
    def gaussian_kernel_3x3(self, sigma):
        device = sigma.device
        C = sigma.shape[1]  # Number of channels
        # 1D coordinate grid
        coords = torch.tensor([-1.0, 0.0, 1.0], device=device)
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")  # Shape: [3, 3]
        # Compute 3x3 Gaussian kernel
        sigma_sq = self.get_sigma(sigma) ** 2  # Shape: [1, C, 1, 1]
        kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma_sq))  # Broadcasted
        # Normalize
        kernel = kernel / kernel.sum(dim=[0, 1], keepdim=True)  
        return kernel.view(C, 1, 3, 3)

    def smooth_with_gaussian(self, x, sigma):
        B, C, H, W = x.shape
        # Ensure sigma has the right shape
        if sigma.shape != (1, C, 1, 1):
            sigma = sigma.view(1, C, 1, 1)
        kernel = self.gaussian_kernel_3x3(sigma)  # [C, 1, 3, 3]
        padding = 1  # To maintain spatial dimensions
        smoothed_x = F.conv2d(x, kernel, groups=C, padding=padding)
        return smoothed_x
    
    def apply_dog(self, x):
        x_smoother = self.smooth_with_gaussian(x, self.sigma_raw_1)
        x_smoothest = self.smooth_with_gaussian(x, self.sigma_raw_2)
        return x_smoother - x_smoothest
    
    def forward(self, skip, dec):
        if self.not_use_this:
            return skip
        y = dec+skip if self.mode=='add' else torch.cat([dec, skip], dim=1)
        x_fea = self.boundary(y)+y if self.use_fea else 0
        x_dog = self.apply_dog(y)+y if self.use_dog else 0
        if self.do_seq:
            y = x_fea + x_dog if self.use_fea or self.use_dog else y
            x_fea = x_dog = 0
        x_dat = self.apply_diffattn(y) if self.use_diffattn else 0
        z = x_fea + x_dog + x_dat if self.use_fea or self.use_dog or self.use_diffattn else y
        z = self.mixer(z)
        return z + skip
