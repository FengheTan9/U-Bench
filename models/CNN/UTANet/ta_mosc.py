
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, List, Set

class Expert(nn.Module):
    """
    Expert module consisting of 1x1 convolutional layers with ReLU activation.
    
    Args:
        emb_size (int): Input embedding size.
        hidden_rate (int, optional): Multiplier for hidden layer size. Defaults to 2.
    """
    def __init__(self, emb_size: int, hidden_rate: int = 2):
        super().__init__()
        hidden_emb = hidden_rate * emb_size
        self.seq = nn.Sequential(
            nn.Conv2d(emb_size, hidden_emb, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(hidden_emb, hidden_emb, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hidden_emb),
            nn.ReLU(),
            nn.Conv2d(hidden_emb, emb_size, kernel_size=1, stride=1, padding=0, bias=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the expert network."""
        return self.seq(x)

class MoE(nn.Module):
    """
    Mixture of Experts (MoE) module with multiple gating mechanisms.
    
    Args:
        num_experts (int): Number of expert networks.
        top (int, optional): Number of top experts to select. Defaults to 2.
        emb_size (int, optional): Embedding dimension. Defaults to 128.
        H (int, optional): Input height. Defaults to 224.
        W (int, optional): Input width. Defaults to 224.
    """
    def __init__(self, num_experts: int, top: int = 2, emb_size: int = 128, H: int = 224, W: int = 224):
        super().__init__()
        self.experts = nn.ModuleList([Expert(emb_size) for _ in range(num_experts)])
        self.gate1 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self.gate2 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self.gate3 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self.gate4 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self._initialize_weights()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.top = top
        
    def _initialize_weights(self) -> None:
        """Initialize gate weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.gate1)
        nn.init.xavier_uniform_(self.gate2)
        nn.init.xavier_uniform_(self.gate3)
        nn.init.xavier_uniform_(self.gate4)
        
    def cv_squared(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the squared coefficient of variation.
        
        Used as a load balancing loss to encourage uniform expert usage.
        
        Args:
            x (torch.Tensor): Tensor of expert usage values.
            
        Returns:
            torch.Tensor: Squared coefficient of variation.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)
        
    def _process_gate(self, x: torch.Tensor, gate_weights: nn.Parameter) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input through a single gating mechanism.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, emb_size).
            gate_weights (nn.Parameter): Gate weights for this gating mechanism.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and load balancing loss.
        """
        batch_size, emb_size, H, W = x.shape
        
        # Compute gating probabilities
        x0 = self.gap(x).view(batch_size, emb_size)
        gate_out = F.softmax(x0 @ gate_weights, dim=1)
        
        # Calculate expert usage for load balancing
        expert_usage = gate_out.sum(0)
        
        # Select top-k experts
        top_weights, top_index = torch.topk(gate_out, self.top, dim=1)
        used_experts = torch.unique(top_index)
        unused_experts = set(range(len(self.experts))) - set(used_experts.tolist())
        
        # Apply softmax again for normalized weights
        top_weights = F.softmax(top_weights, dim=1)
        
        # Expand input for parallel expert processing
        x_expanded = x.unsqueeze(1).expand(batch_size, self.top, emb_size, H, W).reshape(-1, emb_size, H, W)
        y = torch.zeros_like(x_expanded)
        
        # Process each expert
        for expert_i, expert_model in enumerate(self.experts):
            expert_mask = (top_index == expert_i).view(-1)
            expert_indices = expert_mask.nonzero().flatten()
            
            if expert_indices.numel() > 0:
                x_expert = x_expanded[expert_indices]
                y_expert = expert_model(x_expert)
                y = y.index_add(dim=0, index=expert_indices, source=y_expert)
            elif expert_i in unused_experts and self.training:
                # Ensure all experts are used during training
                random_sample = torch.randint(0, x.size(0), (1,), device=x.device)
                x_expert = x_expanded[random_sample]
                y_expert = expert_model(x_expert)
                y = y.index_add(dim=0, index=random_sample, source=y_expert)
        
        # Apply weights and reshape
        top_weights = top_weights.view(-1, 1, 1, 1).expand_as(y)
        y = y * top_weights
        y = y.view(batch_size, self.top, emb_size, H, W).sum(dim=1)
        
        return y, self.cv_squared(expert_usage)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through all gating mechanisms.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, emb_size, H, W).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                Four output tensors and combined load balancing loss.
        """
        #import pdb;pdb.set_trace()
        y1, loss1 = self._process_gate(x, self.gate1)
        y2, loss2 = self._process_gate(x, self.gate2)
        y3, loss3 = self._process_gate(x, self.gate3)
        y4, loss4 = self._process_gate(x, self.gate4)
        
        # Combine losses
        loss = loss1 + loss2 + loss3 + loss4
        
        #if self.training:
            #print(f"Expert Usage - Gate1: {self._format_usage([loss1,loss])}")
            #print(f"Expert Usage - Gate2: {self._format_usage([loss2,loss])}")
            #print(f"Expert Usage - Gate3: {self._format_usage([loss3,loss])}")
            #print(f"Expert Usage - Gate4: {self._format_usage([loss4,loss])}")
        
        return y1, y2, y3, y4, loss
    
    def _format_usage(self, usage: torch.Tensor) -> str:
        """Format expert usage statistics for logging."""
        return f"Min: {usage.min():.4f}, Max: {usage.max():.4f}, CV²: {self.cv_squared(usage):.4f}"

def count_parameters(model: nn.Module) -> str:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model.
        
    Returns:
        str: Formatted string with parameter count.
    """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if params >= 1e6:
        return f"{params / 1e6:.2f}M parameters"
    elif params >= 1e3:
        return f"{params / 1e3:.2f}K parameters"
    else:
        return f"{params} parameters"

if __name__ == '__main__':
    """Unit test for MoE module."""
    try:
        # Initialize model
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MoE(num_experts=4, top=2, emb_size=128, H=224, W=224).to(device)
        model.train()
        
        # Generate random input
        emb = torch.randn(6, 128, 224, 224).to(device)
        
        # Forward pass
        out1, out2, out3, out4, loss = model(emb)
        
        # Verify output shapes
        assert out1.shape == emb.shape, f"Output shape mismatch: {out1.shape} vs {emb.shape}"
        assert out2.shape == emb.shape, f"Output shape mismatch: {out2.shape} vs {emb.shape}"
        assert out3.shape == emb.shape, f"Output shape mismatch: {out3.shape} vs {emb.shape}"
        assert out4.shape == emb.shape, f"Output shape mismatch: {out4.shape} vs {emb.shape}"
        
        print("\n=== MoE Module Test Passed ===")

        print(f"Input Shape: {emb.shape}")
        print(f"Output Shapes: {out1.shape}, {out2.shape}, {out3.shape}, {out4.shape}")
        print(f"Load Balancing Loss: {loss.item():.4f}")
        print(f"Model Parameters: {count_parameters(model)}")
        
    except Exception as e:
        print(f"Test failed: {e}")