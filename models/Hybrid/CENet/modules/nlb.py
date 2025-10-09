

import torch  
import torch.nn as nn  
import torch.nn.functional as F  


class Nonlocal(nn.Module):
    """
    Builds Non-local Neural Networks as a generic family of building
    blocks for capturing long-range dependencies. Non-local Network
    computes the response at a position as a weighted sum of the
    features at all positions. This building block can be plugged into
    many computer vision architectures.
    More details in the paper: https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(
        self,
        dim_inner,
        pool_size=None,
        instantiation="softmax",
        zero_init_final_conv=False,
        zero_init_final_norm=True,
        norm_eps=1e-5,
        norm_momentum=0.1,
        norm_module=nn.BatchNorm2d,
    ):
        """
        Args:
            dim (int): number of dimension for the input.
            dim_inner (int): number of dimension inside of the Non-local block.
            pool_size (list): the kernel size of spatial temporal pooling,
                temporal pool kernel size, spatial pool kernel size, spatial
                pool kernel size in order. By default pool_size is None,
                then there would be no pooling used.
            instantiation (string): supports two different instantiation method:
                "dot_product": normalizing correlation matrix with L2.
                "softmax": normalizing correlation matrix with Softmax.
            zero_init_final_conv (bool): If true, zero initializing the final
                convolution of the Non-local block.
            zero_init_final_norm (bool):
                If true, zero initializing the final batch norm of the Non-local
                block.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(Nonlocal, self).__init__()
        # self.dim = dim
        self.dim_inner = dim_inner
        self.pool_size = pool_size
        self.instantiation = instantiation
        self.use_pool = (
            False if pool_size is None else any((size > 1 for size in pool_size))
        )
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self._construct_nonlocal(
            zero_init_final_conv, zero_init_final_norm, norm_module
        )

    def _construct_nonlocal(
        self, zero_init_final_conv, zero_init_final_norm, norm_module
    ):
        # Three convolution heads: theta, phi, and g.
        self.conv_theta = nn.Conv2d(
            self.dim_inner, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_phi = nn.Conv2d(
            self.dim_inner, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_g = nn.Conv2d(
            self.dim_inner, self.dim_inner, kernel_size=1, stride=1, padding=0
        )

        # Final convolution output.
        self.conv_out = nn.Conv2d(
            self.dim_inner, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        # Zero initializing the final convolution output.
        self.conv_out.zero_init = zero_init_final_conv

        # TODO: change the name to `norm`
        self.bn = norm_module(
            num_features=self.dim_inner,
            eps=self.norm_eps,
            momentum=self.norm_momentum,
        )
        # Zero initializing the final bn.
        self.bn.transform_final_bn = zero_init_final_norm
        
        self.w = nn.Parameter(torch.tensor(0.5))  # Initial weight

        # Optional to add the spatial-temporal pooling.
        if self.use_pool:
            self.pool = nn.MaxPool2d(
                kernel_size=self.pool_size,
                stride=self.pool_size,
                padding=[0, 0],
            )

    def forward(self, x):
        x_identity = x.clone()
        N, C, H, W = x.size()

        theta = self.conv_theta(x)

        # print(theta.shape)
        # Perform temporal-spatial pooling to reduce the computation.
        if self.use_pool:
            x = self.pool(x)

        phi = self.conv_phi(x)
        g = self.conv_g(x)

        theta = theta.view(N, self.dim_inner, -1)
        phi = phi.view(N, self.dim_inner, -1)
        g = g.view(N, self.dim_inner, -1)

        # (N, C, HxW) * (N, C, HxW) => (N, HxW, HxW).
        theta_phi = torch.einsum("nch,ncp->nhp", (theta, phi))
        # print(theta_phi.shape)
        # For original Non-local paper, there are two main ways to normalize
        # the affinity tensor:
        #   1) Softmax normalization (norm on exp).
        #   2) dot_product normalization.
        if self.instantiation == "softmax":
            # Normalizing the affinity tensor theta_phi before softmax.
            theta_phi = theta_phi * (self.dim_inner**-0.5)
            theta_phi = nn.functional.softmax(theta_phi, dim=2)
        elif self.instantiation == "dot_product":
            spatial_dim = theta_phi.shape[2]
            theta_phi = theta_phi / spatial_dim
        else:
            raise NotImplementedError("Unknown norm type {}".format(self.instantiation))

        # (N, HW, HW) * (N, C, HW) => (N, C, HW)  
        theta_phi_g = torch.einsum("nhg,ncg->nch", (theta_phi, g))
        
        # print(theta_phi_g.shape)
        # (N, C, HxW) => (N, C, H, W).
        theta_phi_g = theta_phi_g.view(N, self.dim_inner, H, W)

        p = self.conv_out(theta_phi_g)
        p = self.bn(p)
        
        z = (1 - self.w) * x_identity + self.w * p
        return z


class WeightedNonLocalBlock(nn.Module):  
    def __init__(self, input_channels):  
        super(WeightedNonLocalBlock, self).__init__()  

        # Point-wise convolutions for theta, phi, and g - maintaining input channels  
        self.theta_conv = nn.Conv2d(input_channels, input_channels, kernel_size=1)  
        self.phi_conv = nn.Conv2d(input_channels, input_channels, kernel_size=1)  
        self.g_conv = nn.Conv2d(input_channels, input_channels, kernel_size=1)  

        # Learnable weight (scalar)  
        self.w = nn.Parameter(torch.tensor(0.5))  # Initial weight  

        # Final point-wise convolution for output feature map  
        self.Wz_conv = nn.Conv2d(input_channels, input_channels, kernel_size=1)  

    def forward(self, x):  
        # Step 1: Compute theta, phi, and g using convolutions  
        theta_x = self.theta_conv(x)  # (N, C, H, W)  
        phi_x = self.phi_conv(x)      # (N, C, H, W)  
        g_x = self.g_conv(x)          # (N, C, H, W)  

        # Step 2: Reshape tensors to prepare for attention computation  
        batch_size, channels, height, width = x.size()  
        
        theta_x = theta_x.view(batch_size, channels, -1)  # (N, C, H*W)  
        phi_x = phi_x.view(batch_size, channels, -1)      # (N, C, H*W)  
        g_x = g_x.view(batch_size, channels, -1)          # (N, C, H*W)  

        # Step 3: Compute the attention map  
        attention_map = F.softmax(torch.bmm(theta_x.transpose(1, 2), phi_x), dim=-1)  # (N, H*W, H*W)  

        # Step 4: Compute the non-local operation  
        y = torch.bmm(attention_map, g_x.transpose(1, 2))  # (N, H*W, C)  

        # Step 5: Reshape y back to spatial dimensions  
        y = y.view(batch_size, channels, height, width)  # (N, C, H, W)  

        # Step 6: Combine the output with the original input  
        z = (1 - self.w) * x + self.w * self.Wz_conv(y)  # Apply final convolution  

        return z  

# Example usage  
if __name__ == "__main__":  
    input_tensor = torch.randn(8, 64, 32, 32)  # Example input: (batch_size, channels, height, width)  
    model = WeightedNonLocalBlock(input_channels=64)  
    output = model(input_tensor)  
    print(output.shape)  # Check output shape

