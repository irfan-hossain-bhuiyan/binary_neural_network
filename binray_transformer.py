from math import log
import copy
from numpy import random
import torch
from torch._prims_common import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Callable, cast
from torch.optim import Adam, RAdam
from prelude import DEVICE, TrainerState,  leaky_clamp, Trainer, plot_training_loss, split_dataset, stop_on_epoch
from data_utils import load_mnist, save_xor_dataset, load_xor_dataset

def xor(a:Tensor,b:Tensor)->torch.Tensor:
    return a+b-2*a*b
class OrNorGateLayer(nn.Module):
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
        self.bias = nn.Parameter(torch.ones(out_features, in_features))
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

    def actual_bias(self) -> torch.Tensor:
        return cast(torch.Tensor, leaky_clamp(self.bias, 0, 1, 0.1))

    def discretize(self, threshold: float) -> None:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        with torch.no_grad():
            # Evaluation of clamp_max_ and clamp_min_ in-place would modify the whole tensor unpredictably.
            # Using out-of-place equivalents:
            # For weights < threshold, ensure they are <= 0 (clamp_max(..., 0))
            # For weights >= threshold, ensure they are >= 1 (clamp_min(..., 1))
            discrete_w = torch.where(
                self.weight < threshold,
                torch.clamp_max(self.weight, 0),
                torch.clamp_min(self.weight, 1)
            )
            self.weight.copy_(discrete_w)
            
            discrete_b = torch.where(
                self.bias < threshold,
                torch.clamp_max(self.bias, 0),
                torch.clamp_min(self.bias, 1)
            )
            self.bias.copy_(discrete_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        actual_weight = self.actual_weight()
        actual_bias = self.actual_bias()
        # Input tensor expanded for operations
        x_exp = x.unsqueeze(1)
        b_exp = actual_bias.unsqueeze(0)
        # Continuous XOR: x * (1 - b) + b * (1 - x) = x + b - 2*x*b
        x_xor_b = x_exp + b_exp - 2.0 * x_exp * b_exp
        
        # z: (batch_size, out_features, in_features)
        z = x_xor_b * actual_weight.unsqueeze(0)
        
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
                    s.register_hook(lambda grad: grad / (max_p_s + 2e-1))
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
        even_initialization: Callable[..., Any] =lambda x:nn.init.normal_(x,mean=1.0),
        odd_initialization:None | Callable[...,Any] =lambda x:nn.init.normal_(x,mean=0.0),
        grad_scalar:bool=False,
        load_file: str | Path | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.layer_dims = list(layer_dims)
        self.use_softmax = use_softmax
        self.is_shared_tau = isinstance(init_tau_param,nn.Parameter)
        self.expectation_layers: nn.ModuleList = nn.ModuleList()
        if odd_initialization is None:
            odd_initialization=even_initialization
        in_dim = input_dim # As the first one passes to an inverter.
        for i,out_dim in enumerate(self.layer_dims):
            initialization=even_initialization if i%2==0 else odd_initialization
            layer = OrNorGateLayer(
                in_features=in_dim,
                out_features=out_dim,
                tau=init_tau_param,
                use_softmax=use_softmax,
                max_threshold=max_threshold,
                initialization=initialization,
                grad_scalar=grad_scalar,
            )
            self.expectation_layers.append(layer)
            in_dim = out_dim

        if load_file is not None:
            load_path = Path(load_file) if isinstance(load_file, str) else load_file
            if load_path.exists():
                self.load_state_dict(torch.load(load_path, weights_only=True))
            else:
                # Ensure parent directory exists
                load_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.state_dict(), load_path)
 
    def clone(self):
        return copy.deepcopy(self)

    def regularization_factory(l1_lambda=1e-1, disc_lambda=1e-1, tau_lambda=1e-1,default:bool=True,error_scaler_on_platau:None|float=None):
        error_scale=1
        def regularization(module:"MultiLayerLogicGateNet",state:TrainerState):
            nonlocal error_scale
            if error_scaler_on_platau is not None:
                if state.is_plateaued: error_scale*=error_scaler_on_platau
            reg = torch.tensor(0.0, device=DEVICE)
            for layer in module.expectation_layers:
                layer = cast(OrNorGateLayer, layer)
                w = layer.weight
                b = layer.bias
                
                l1_error = w.relu().mean() 
                disc_error_w = (0.5-(w-0.5).abs()).relu().mean()
                disc_error_b = (0.5-(b-0.5).abs()).relu().mean()
                disc_error = (disc_error_w + disc_error_b)
                
                tau_err = torch.exp(-layer.tau)
                reg += (l1_lambda * l1_error) + (disc_lambda * disc_error) + (tau_lambda * tau_err)
                # Encourage tau to grow larger (L1 regularization, negative sign)
            return reg*error_scale
        def close_to_discrete(module:Any):
            reg = torch.tensor(0.0, device=DEVICE)
            for layer in module.expectation_layers:
                layer = cast(OrNorGateLayer, layer)
                w = layer.weight
                b = layer.bias
                disc_error_w = (0.5-(w-0.5).abs()).abs().mean()
                disc_error_b = (0.5-(b-0.5).abs()).abs().mean()
                disc_error=disc_error_w+disc_error_b
                tau_err = torch.exp(-layer.tau)
                reg+=disc_lambda*disc_error+tau_lambda*tau_err
            return reg*random.rand()
        if default:return regularization
        else:return close_to_discrete
    def set_use_softmax(self,value:bool):
        for layer in self.expectation_layers:
            layer = cast(OrNorGateLayer,layer)
            layer.use_softmax=value

    def peek(self) -> dict[str, Any]:
        result = {}
        with torch.no_grad():
            if self.is_shared_tau:
                first_layer = cast(OrNorGateLayer, self.expectation_layers[0])
                result["shared_tau"] = first_layer.tau.mean().item() if first_layer.tau.numel() > 1 else first_layer.tau.item()
            else:
                for i, layer in enumerate(self.expectation_layers):
                    layer = cast(OrNorGateLayer, layer)
                    result[f"tau_{i}"] = layer.tau.mean().item() if layer.tau.numel() > 1 else layer.tau.item()
        return result

    def constraint(module:Any):
        for layer in module.expectation_layers:
            layer = cast(OrNorGateLayer, layer)
            layer.weight.clamp_(-10.0, 10.0)
            layer.bias.clamp_(-10.0, 10.0)
            layer.tau_costraint(20)


    @property
    def tau(self) -> torch.Tensor | list[torch.Tensor]:
        if self.is_shared_tau:
            first_layer = cast(OrNorGateLayer, self.expectation_layers[0])
            return first_layer.tau
        return [cast(OrNorGateLayer, layer).tau for layer in self.expectation_layers]

    def discretize(module:Any, threshold: float=0.5) -> None:
        for layer in module.expectation_layers:
            layer = cast(OrNorGateLayer, layer)
            layer.discretize(threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.expectation_layers):
            x = layer(x)
        return x

def train_mnist():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path("artifacts/mnist_binary.pt")
    x_all, y_all = load_mnist(dataset_path, device=device, input_flatten=True)
    x_train, y_train, _, _ = split_dataset(x_all, y_all, train_ratio=0.8, shuffle=True)
    
    # Base network to clone from, ensuring all models start with the same exact weights
    base_net = MultiLayerLogicGateNet(
        input_dim=784,
        layer_dims=(128,64,128,64, 128, 4),
        use_softmax=True,
    )
    
    
def train_xor_extend_layer(epoch:int=50,is_dataparallel:bool=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path("artifacts/xor_dataset.pt")
    if not dataset_path.exists():
        save_xor_dataset(dataset_path, num_samples=100000)

    x_all, y_all = load_xor_dataset(dataset_path, device=device)
    x_train, y_train, _, _ = split_dataset(x_all, y_all, train_ratio=0.8, shuffle=True)

    base_net = MultiLayerLogicGateNet(
        input_dim=64,
        layer_dims=(256, 128, 64,128,64 ,32),
        use_softmax=True,
        max_threshold=0.95,
    )
    from prelude import stop_on_epoch
    import matplotlib.pyplot as plt
    
    l1_regs = [0.01, 0.05, 0.1, 0.5]
    checkpoints = []

    for l1 in l1_regs:
        print(f"\n{'='*50}\nStarting training with L1 = {l1}, Disc = {l1}, Tau = {l1}\n{'='*50}")
        net = base_net.clone()
        
        if torch.cuda.device_count() > 1 and is_dataparallel:
            print(f"Using {torch.cuda.device_count()} GPUs for MNIST!")
            model = nn.DataParallel(net)
        else:
            model = net

        trainer = Trainer(
            dataset=(x_train, y_train),
            stop_on=stop_on_epoch(epoch),
            batch_size=128,
            model=model,
            loss_fn=nn.HuberLoss(delta=0.5),
            optimizer_cls=torch.optim.RAdam,
            optimizer_kwargs={},
            regularization_fn= MultiLayerLogicGateNet.regularization_factory(l1, l1, l1),
            lr_scheduler_factory=None,#fn_call_on_plateau_scheduler(MultiLayerLogicGateNet.discretize),
            constraint=MultiLayerLogicGateNet.constraint,
            checkpoint_path=None, # Don't overwrite for each run
            device=device,
            check_grad=False, # Turned off to reduce console spam for 4 runs
            state=TrainerState(50),
            peek=net.peek,
        )
        ckpt = trainer.train()
        checkpoints.append((l1, ckpt))

    # Plot all errors together
    plt.figure(figsize=(10, 6))
    for l1, ckpt in checkpoints:
        errors = ckpt.avg_errors()
        plt.plot(range(1, len(errors) + 1), errors, label=f"L1 = {l1}", linewidth=2)
        
    plt.xlabel("Epoch")
    plt.ylabel("Testing Error")
    plt.title("Effect of L1 Regularization on Training Error")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return checkpoints

def train_xor_main(run_id="", epoch=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path("artifacts/xor_dataset.pt")
    if not dataset_path.exists():
        save_xor_dataset(dataset_path, num_samples=100000)

    x_all, y_all = load_xor_dataset(dataset_path, device=device)
    x_train, y_train, _, _ = split_dataset(x_all, y_all, train_ratio=0.8, shuffle=True)

    net = MultiLayerLogicGateNet(
        input_dim=64,
        layer_dims=(256, 128, 64,128,64,32),
        use_softmax=True,
        max_threshold=0.95,
        grad_scalar=True,
    )        
    trainer = Trainer(
        dataset=(x_train, y_train),
        stop_on=stop_on_epoch(epoch),
        batch_size=128,
        model=net,
        loss_fn=nn.MSELoss(),
        optimizer_cls=Adam,
        optimizer_kwargs={},
        regularization_fn= MultiLayerLogicGateNet.regularization_factory(0.4,0.4,0.5),
        lr_scheduler_factory=None,#fn_call_on_plateau_scheduler(MultiLayerLogicGateNet.discretize),
        constraint=MultiLayerLogicGateNet.constraint,
        checkpoint_path=None, # Don't overwrite for each run
        device=device,
        check_grad=False, # Turned off to reduce console spam for 4 runs
        state=TrainerState(50),
        peek=net.peek,
    )
    ckpt = trainer.train()
    plot_training_loss(ckpt.avg_errors(), header=f"Run {run_id}" if run_id else "")
    return ckpt

def run_train_xor_main_sequential():
    print("Running train_xor_main 3 times sequentially...")
    results = []
    
    for i in range(1, 4):
        print(f"\n========== STARTING RUN {i} ==========")
        try:
            res = train_xor_main(run_id=str(i))
            results.append(res)
            print(f"========== COMPLETED RUN {i} ==========")
        except Exception as e:
            print(f"========== FAILED RUN {i}: {e} ==========")
            
    return results

def main():
    run_train_xor_main_sequential()
if __name__ == "__main__":
    main()

