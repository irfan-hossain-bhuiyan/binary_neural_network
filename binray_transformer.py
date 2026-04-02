from math import log
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Callable, cast
from torch.optim import Adam
from prelude import DEVICE, animate_gradient_flow, discretize_on_plateau_scheduler, early_stopping,load_checkpoint, plateau_scheduler, stop_on_epoch, leaky_clamp, plot_training_loss, Trainer, split_dataset
from data_utils import load_mnist, save_xor_dataset, load_xor_dataset
from prelude import plot_weight_distribution

def pass_invert(x: torch.Tensor) -> torch.Tensor:
    """Concatenate inputs with their inverted values (1 - x)."""
    inverted = 1.0 - x
    return torch.cat([x, inverted], dim=-1)

class OrGateLayer(nn.Module):
    """Expectation layer that can operate in soft (softmax) or hard (argmax) mode.

    Args:
        in_features: input feature dimension
        out_features: output feature dimension
        shared_tau_unconstrained: if a float is provided (default), a new nn.Parameter is created
            and owned by this layer (not shared). If an nn.Parameter is provided, it
            will be used directly, enabling temperature sharing across layers.
        use_softmax: if True, uses temperature-scaled softmax expectation; otherwise
            uses hard max selection.
        max_threshold: Used when softmax is used,It make a upper floor for the softmax function,Say if 
        in or gate rest of the input are 0 and one is 1,(output should be 1),but softmax doesn't make that,
        This parameter sets what is the minimum truth value will be in that scenerio.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_threshold: float = 0.9,
        tau: float|nn.Parameter = 0.0,
        use_softmax: bool = False,
        initialization: Callable[..., Any] = nn.init.normal_,
        grad_scalar:bool=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_softmax = use_softmax
        self.grad_scalar=grad_scalar
        # Compute gradient scale based on square root of the input dimension

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        initialization(self.weight)
        if isinstance(tau,nn.Parameter):
            self.tau_adder=tau
        else:
            self.tau_adder=nn.Parameter(torch.tensor(tau))
        self.tau_floor=log(in_features-1)+log(max_threshold)-log(1-max_threshold)
    @property
    def tau(self) -> torch.Tensor:
        return self.tau_floor + F.leaky_relu(self.tau_adder,negative_slope=0.05)
    
    def tau_costraint(self,max_value):
        self.tau_adder.clamp_max_(max_value)

    def actual_weight(self) -> torch.Tensor:
        return cast(torch.Tensor, leaky_clamp(self.weight, 0, 1, 0.1))

    def discretize(self, threshold: float) -> None:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        with torch.no_grad():
            discrete_w = torch.where(self.weight<threshold,torch.clamp_min_(self.weight,0),torch.clamp_max_(self.weight,1))#(self.weight >= threshold).float()
            self.weight.copy_(discrete_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        actual_weight = self.actual_weight()
        # z: (batch_size, out_features, in_features)
        z = x.unsqueeze(1) * actual_weight.unsqueeze(0)
        
        if self.use_softmax:
            z_scaled = self.tau * z
            p = F.softmax(z_scaled, dim=-1)
            s = (p * z).sum(dim=-1)
            
            # Hook on z: scales gradient AFTER passing backward through softmax/sum gate
            #if z.requires_grad:
                # keepdim=True leaves the last dimension as 1...
            #    max_p = p.max(dim=-1, keepdim=True).values.detach()
                # Use default args lambda grad, mp=max_p to FORCE capture by value 
                # (prevents lambda from capturing max_p of other layers due to late binding)
            #    z.register_hook(lambda grad, mp=max_p: grad / (mp + 1e-4))
                
            # Hook on s: scales gradient BEFORE passing backward through the gate
            if self.grad_scalar:
                if s.requires_grad:
                    max_p_s = p.max(dim=-1).values.detach()
                    s.register_hook(lambda grad: grad / (max_p_s + 1e-1))
        else:
            s = z.max(dim=-1).values

        return s
    def to_hardmax(self):
        self.use_softmax=False

class MultiLayerLogicGateNet(nn.Module):
    """Expectation-based multi-layer gate network with configurable depth and tau sharing."""

    def __init__(
        self,
        input_dim: int = 64,
        layer_dims: list[int] | tuple[int, ...] = (256, 128, 64, 32),
        init_tau_param:nn.Parameter |float = 0.0,
        max_threshold:float =0.95,
        use_softmax: bool = False,
        only_inverter=True,
        even_initialization: Callable[..., Any] =lambda x:nn.init.normal_(x,mean=1.0),
        odd_initialization:None | Callable[...,Any] =lambda x:nn.init.normal_(x,mean=0.0),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.layer_dims = list(layer_dims)
        self.use_softmax = use_softmax
        self.is_shared_tau = isinstance(init_tau_param,nn.Parameter)
        self.only_inverter=only_inverter
        self.expectation_layers: nn.ModuleList = nn.ModuleList()
        if odd_initialization is None:
            odd_initialization=even_initialization
        in_dim = input_dim *2 # As the first one passes to an inverter.
        for i,out_dim in enumerate(self.layer_dims):
            initialization=even_initialization if i%2==0 else odd_initialization
            layer = OrGateLayer(
                in_features=in_dim,
                out_features=out_dim,
                tau=init_tau_param,
                use_softmax=use_softmax,
                max_threshold=max_threshold,
                initialization=initialization,
            )
            self.expectation_layers.append(layer)
            in_dim = out_dim * (1 if only_inverter else 2) # inverter doubles the features
 
    def clone(self):
        return copy.deepcopy(self)

        
    def regularization(module:Any, l1_lambda=1e-1, disc_lambda=1e-1, tau_lambda=1e-1):
        reg = torch.tensor(0.0, device=DEVICE)
        for layer in module.expectation_layers:
            layer = cast(OrGateLayer, layer)
            w = layer.weight
            l1_error = w.relu().mean()
            disc_error = (0.5-(w-0.5).abs()).relu().mean()
            tau_err = torch.exp(-layer.tau)
            reg += (l1_lambda * l1_error) + (disc_lambda * disc_error) + (tau_lambda * tau_err)
            # Encourage tau to grow larger (L1 regularization, negative sign)
        return reg
    def set_use_softmax(self,value:bool):
        for layer in self.expectation_layers:
            layer = cast(OrGateLayer,layer)
            layer.use_softmax=value

    def peek(self) -> dict[str, Any]:
        result = {}
        with torch.no_grad():
            if self.is_shared_tau:
                first_layer = cast(OrGateLayer, self.expectation_layers[0])
                result["shared_tau"] = first_layer.tau.mean().item() if first_layer.tau.numel() > 1 else first_layer.tau.item()
            else:
                for i, layer in enumerate(self.expectation_layers):
                    layer = cast(OrGateLayer, layer)
                    result[f"tau_{i}"] = layer.tau.mean().item() if layer.tau.numel() > 1 else layer.tau.item()
        return result
    
    def constraint(module:Any):
        for layer in module.expectation_layers:
            layer = cast(OrGateLayer, layer)
            layer.weight.clamp_(-20.0, 20.0)
            layer.tau_costraint(20)


    @property
    def tau(self) -> torch.Tensor | list[torch.Tensor]:
        if self.is_shared_tau:
            first_layer = cast(OrGateLayer, self.expectation_layers[0])
            return first_layer.tau
        return [cast(OrGateLayer, layer).tau for layer in self.expectation_layers]

    def discretize(self, threshold: float=0.5) -> None:
        for layer in self.expectation_layers:
            layer = cast(OrGateLayer, layer)
            layer.discretize(threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = pass_invert(x)
        for idx, layer in enumerate(self.expectation_layers):
            x = layer(x)
            if idx < len(self.expectation_layers) - 1:
                x = (1-x) if self.only_inverter else pass_invert(x)
            else:
                x = (1-x)
        return x

def train_mnist(save_checkpoint: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path("artifacts/mnist_binary.pt")
    x_all, y_all = load_mnist(dataset_path, device=device, input_flatten=True)
    x_train, y_train, _, _ = split_dataset(x_all, y_all, train_ratio=0.8, shuffle=True)
    net = MultiLayerLogicGateNet(
        input_dim=784,
        layer_dims=(128,64,128,64, 128, 4),
        only_inverter=True,
        use_softmax=True,
    #even_initialization=lambda x:nn.init.normal_(x,mean=0),
    #odd_initialization=lambda x:nn.init.normal_(x,mean=1)
    )
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for MNIST!")
        model = nn.DataParallel(net)
    else:
        model = net

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    trainer = Trainer(
        dataset=(x_train, y_train),
        stop_on=early_stopping(30, max_epochs=100),
        batch_size=128,
        model=model,
        loss_fn=nn.HuberLoss(delta=0.5),
        optimizer_cls=Adam,
        optimizer_kwargs={"betas": (0.5, 0.5), "lr": 1},
        regularization_fn=None,
        lr_scheduler_factory=discretize_on_plateau_scheduler(),
        constraint=lambda m: MultiLayerLogicGateNet.constraint(m.module if isinstance(m, nn.DataParallel) else m),
        checkpoint_path=Path("artifacts/mnist_transformer_checkpoint.pt") if save_checkpoint else None,
        device=device,
        check_grad=True,
        peek=net.peek,
    )
    checkpoint = trainer.train()
    plot_training_loss(checkpoint.avg_losses())
    plot_weight_distribution(checkpoint.model)
    return checkpoint

def train_xor(save_checkpoint: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path("artifacts/xor_dataset.pt")
    if not dataset_path.exists():
        save_xor_dataset(dataset_path, num_samples=100000)

    x_all, y_all = load_xor_dataset(dataset_path, device=device)
    x_train, y_train, _, _ = split_dataset(x_all, y_all, train_ratio=0.8, shuffle=True)

    net = MultiLayerLogicGateNet(
        input_dim=64,
        layer_dims=(256, 128, 64, 32),
        use_softmax=True,
        only_inverter=True,
        max_threshold=0.95,
    )
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for XOR!")
        model = nn.DataParallel(net)
    else:
        model = net
    
    
    # We define a custom preview function that both gives string metrics to Console
    # and logs to TensorBoard for plotting.
    trainer = Trainer(
        dataset=(x_train, y_train),
        stop_on=early_stopping(10, max_epochs=200),
        batch_size=128,
        model=model,
        loss_fn=nn.HuberLoss(delta=0.5),
        optimizer_cls= Adam,
        optimizer_kwargs= {"betas":(0.25,0.25),"lr":0.1},
        regularization_fn=lambda :net.regularization(1e-1,1e-1,1e-1),
        lr_scheduler_factory= discretize_on_plateau_scheduler(),
        constraint=lambda m: MultiLayerLogicGateNet.constraint(m.module if isinstance(m, nn.DataParallel) else m),
        checkpoint_path=Path("artifacts/binary_transformer_checkpoint.pt") if save_checkpoint else None,
        device=device,
        check_grad=True,
        peek=net.peek,
    )
    checkpoint = trainer.train()
    #trainer.export_for_burn(Path("artifacts/burn_export"))
    plot_training_loss(checkpoint.avg_errors())
    plot_weight_distribution(checkpoint.model)
    
    # Cleanup TensorBoard
    return checkpoint



def main():
    train_mnist(save_checkpoint=True)
    #checkpoint=load_training_checkpoint("./binary_transformer_checkpoint.pt",DEVICE)
    #animate_gradient_distributions(checkpoint)    
if __name__ == "__main__":
    main()

