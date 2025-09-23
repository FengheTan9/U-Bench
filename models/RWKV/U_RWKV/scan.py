import torch
from einops import rearrange

# def horizontal_forward_scan(input_tensor):
#     """
#     对输入张量进行水平正向扫描
#     输入形状: (B, C, H, W)
#     变换后形状: (B, H * W, C)
#     """
#     B, C, H, W = input_tensor.shape
#     input_tensor = input_tensor.permute(0, 2, 3, 1)  # (B, H, W, C)
#     flattened = input_tensor.reshape(B, H * W, C)  # (B, H * W, C)
#     return flattened.permute(0, 2, 1)  # (B, C, H * W)

# def horizontal_forward_scan_inv(transformed_tensor, original_shape):
#     """
#     逆变换: 复原 horizontal_forward_scan 变换后的数据
#     输入形状: (B, C, H * W)
#     复原形状: (B, C, H, W)
#     """
#     B, C, H, W = original_shape
#     transformed_tensor = transformed_tensor.permute(0, 2, 1)  # (B, H * W, C)
#     recovered = transformed_tensor.view(B, H, W, C)  # 复原为 (B, H, W, C)
#     return recovered.permute(0, 3, 1, 2)  # (B, C, H, W)


# def horizontal_backward_scan(input_tensor):
#     """
#     对输入张量进行水平反向扫描
#     """
#     B, C, H, W = input_tensor.shape
#     input_tensor = torch.flip(input_tensor, dims=[-1])  # 水平翻转 (B, C, H, W)
#     input_tensor = input_tensor.permute(0, 2, 3, 1)  # (B, H, W, C)
#     flattened = input_tensor.reshape(B, H * W, C)  # (B, H * W, C)
#     return flattened.permute(0, 2, 1)  # (B, C, H * W)

# def horizontal_backward_scan_inv(transformed_tensor, original_shape):
#     """
#     逆变换: 复原 horizontal_backward_scan 变换后的数据
#     """
#     B, C, H, W = original_shape
#     transformed_tensor = transformed_tensor.permute(0, 2, 1)  # (B, H * W, C)
#     recovered = transformed_tensor.view(B, H, W, C)  # (B, H, W, C)
#     recovered = recovered.permute(0, 3, 1, 2)  # (B, C, H, W)
#     return torch.flip(recovered, dims=[-1])  # 水平翻转回去

def horizontal_forward_scan(input_tensor):
    """
    对输入张量进行水平正向扫描
    输入形状: (B, C, H, W)
    变换后形状: (B, H * W, C)
    """
    B, C, H, W = input_tensor.shape
    input_tensor = input_tensor.permute(0, 2, 3, 1)  # (B, H, W, C)
    flattened = input_tensor.reshape(B, H * W, C)  # (B, H * W, C)
    return flattened

def horizontal_forward_scan_inv(transformed_tensor, original_shape):
    """
    逆变换: 复原 horizontal_forward_scan 变换后的数据
    输入形状: (B, H * W, C)
    复原形状: (B, C, H, W)
    """
    B, C, H, W = original_shape
    recovered = transformed_tensor.view(B, H, W, C)  # (B, H, W, C)
    return recovered.permute(0, 3, 1, 2)  # (B, C, H, W)


def horizontal_backward_scan(input_tensor):
    """
    对输入张量进行水平反向扫描
    """
    B, C, H, W = input_tensor.shape
    input_tensor = torch.flip(input_tensor, dims=[-1])  # 水平翻转 (B, C, H, W)
    input_tensor = input_tensor.permute(0, 2, 3, 1)  # (B, H, W, C)
    flattened = input_tensor.reshape(B, H * W, C)  # (B, H * W, C)
    return flattened

def horizontal_backward_scan_inv(transformed_tensor, original_shape):
    """
    逆变换: 复原 horizontal_backward_scan 变换后的数据
    """
    B, C, H, W = original_shape
    recovered = transformed_tensor.view(B, H, W, C)  # (B, H, W, C)
    recovered = recovered.permute(0, 3, 1, 2)  # (B, C, H, W)
    return torch.flip(recovered, dims=[-1])  # 水平翻转回去


def vertical_forward_scan(input_tensor):
    """
    对输入张量进行垂直正向扫描
    """
    B, C, H, W = input_tensor.shape
    input_tensor = input_tensor.permute(0, 1, 3, 2)  # (B, C, W, H)
    return input_tensor.flatten(2).permute(0, 2, 1)  # (B, W*H, C)

def vertical_forward_scan_inv(transformed_tensor, original_shape):
    """
    逆变换: 复原 vertical_forward_scan 变换后的数据
    """
    B, C, H, W = original_shape
    transformed_tensor = transformed_tensor.permute(0, 2, 1)  # (B, C, W*H)
    recovered = transformed_tensor.view(B, C, W, H)  # (B, C, W, H)
    return recovered.permute(0, 1, 3, 2)  # (B, C, H, W)


def vertical_backward_scan(input_tensor):
    """
    对输入张量进行垂直反向扫描
    """
    B, C, H, W = input_tensor.shape
    input_tensor = torch.flip(input_tensor, dims=[-2]).contiguous()  # 垂直翻转
    input_tensor = input_tensor.permute(0, 1, 3, 2)  # (B, C, W, H)
    return input_tensor.flatten(2).permute(0, 2, 1)  # (B, W*H, C)

def vertical_backward_scan_inv(transformed_tensor, original_shape):
    """
    逆变换: 复原 vertical_backward_scan 变换后的数据
    """
    B, C, H, W = original_shape
    transformed_tensor = transformed_tensor.permute(0, 2, 1)  # (B, C, W*H)
    recovered = transformed_tensor.view(B, C, W, H)  # (B, C, W, H)
    recovered = recovered.permute(0, 1, 3, 2)  # (B, C, H, W)
    return torch.flip(recovered, dims=[-2])  # 垂直翻转回去

if __name__ == "__main__":
    # 测试代码
    B, C, H, W = 1, 1, 3, 3  # 形状定义
    input_tensor = torch.tensor([[[[1, 2, 3], 
                                [4, 5, 6], 
                                [7, 8, 9]]]], dtype=torch.float32)  # (1,1,3,3)

    # 测试 horizontal_forward_scan
    transformed = horizontal_forward_scan(input_tensor)
    recovered = horizontal_forward_scan_inv(transformed, input_tensor.shape)
    assert torch.allclose(input_tensor, recovered), "horizontal_forward_scan_inv failed!"
    # print(transformed.shape)
    print(transformed)
    import pdb; pdb.set_trace()

    # 测试 horizontal_backward_scan
    transformed = horizontal_backward_scan(input_tensor)
    recovered = horizontal_backward_scan_inv(transformed, input_tensor.shape)
    assert torch.allclose(input_tensor, recovered), "horizontal_backward_scan_inv failed!"

    # 测试 vertical_forward_scan
    transformed = vertical_forward_scan(input_tensor)
    recovered = vertical_forward_scan_inv(transformed, input_tensor.shape)
    assert torch.allclose(input_tensor, recovered), "vertical_forward_scan_inv failed!"

    # 测试 vertical_backward_scan
    transformed = vertical_backward_scan(input_tensor)
    recovered = vertical_backward_scan_inv(transformed, input_tensor.shape)
    assert torch.allclose(input_tensor, recovered), "vertical_backward_scan_inv failed!"

    print("所有变换及其逆变换测试通过！")