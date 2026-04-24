import torch
import torch.nn as nn
from typing import Any, List

class DiscreteOrNorGateLayer(nn.Module):
    """
    Discrete (Boolean) version of OrNorGateLayer.
    Uses native PyTorch boolean tensors and bitwise operations to compute the output.
    Memory efficient and computationally fast since mathematical approximations are avoided.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Registering boolean tensors as buffers so they are serialized in the state_dict
        # but are ignored by PyTorch optimizers (since they are non-differentiable bools)
        self.register_buffer('weight', torch.zeros(out_features, in_features, dtype=torch.bool)) # type: ignore
        self.register_buffer('bias', torch.zeros(out_features, in_features, dtype=torch.bool)) # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using native boolean logic.
        Args:
            x: Input boolean tensor of shape (batch_size, ..., in_features)
        Returns:
            Output boolean tensor of shape (batch_size, ..., out_features)
        """
        # Ensure input is boolean
        x = x.to(torch.bool) # type: ignore
        
        x_exp = x.unsqueeze(-2)      # (batch_size, 1, in_features)
        b_exp = self.bias.unsqueeze(0) # type: ignore
        w_exp = self.weight.unsqueeze(0) # type: ignore
        
        # 1. XOR input with bias (equivalent to: x_exp + b_exp - 2.0*x_exp*b_exp in continuous)
        x_xor_b = x_exp ^ b_exp
        
        # 2. AND with weight (equivalent to: x_xor_b * actual_weight in continuous)
        z = x_xor_b & w_exp
        
        # 3. OR over the input dimension (equivalent to: z.max(dim=-1) in hard continuous mode)
        # .any() computes the logical OR along the given dimension
        s = z.any(dim=-1)
        
        return s


class DiscreteMultiLayerLogicGateNet(nn.Module):
    """
    Strictly Boolean version of MultiLayerLogicGateNet.
    Operates entirely with discrete boolean logic (XOR, AND, OR).
    """
    def __init__(self, input_dim: int, layer_dims: List[int] | tuple[int, ...]):
        super().__init__()
        self.input_dim = input_dim
        self.layer_dims = list(layer_dims)
        
        self.expectation_layers = nn.ModuleList()
        in_dim = input_dim
        for out_dim in self.layer_dims:
            layer = DiscreteOrNorGateLayer(in_features=in_dim, out_features=out_dim)
            self.expectation_layers.append(layer)
            in_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.bool) # type: ignore
        for layer in self.expectation_layers:
            x = layer(x)
        return x

    def to_continuous(self, use_softmax: bool = True, max_threshold: float = 0.95) -> Any:
        """
        Convert this discrete boolean network back into a continuous MultiLayerLogicGateNet.
        The float weights and biases will be perfectly 1.0 or 0.0.
        """
        # Late import to prevent circular import between the two architectures
        from binray_transformer import MultiLayerLogicGateNet
        
        continuous_net = MultiLayerLogicGateNet(
            input_dim=self.input_dim,
            layer_dims=self.layer_dims,
            use_softmax=use_softmax,
            max_threshold=max_threshold,
        )
        
        for cont_layer, disc_layer in zip(continuous_net.expectation_layers, self.expectation_layers):
            # Direct copy from boolean tensor to float tensor (True -> 1.0, False -> 0.0)
            cont_layer.weight.data.copy_(disc_layer.weight.to(torch.float32)) # type: ignore
            cont_layer.bias.data.copy_(disc_layer.bias.to(torch.float32)) # type: ignore
            
        return continuous_net
