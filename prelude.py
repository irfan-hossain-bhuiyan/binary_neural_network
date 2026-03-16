import torch

class LeakyClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min_val, max_val, leak=0.01):
        ctx.save_for_backward(input)
        ctx.min_val = min_val
        ctx.max_val = max_val
        ctx.leak = leak
        return torch.clamp(input, min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Mask where the input was within bounds
        mask = (input >= ctx.min_val) & (input <= ctx.max_val)
        
        # Gradient is 1.0 inside, and 'leak' outside
        grad_input = grad_output.clone()
        grad_input[~mask] *= ctx.leak 
        
        return grad_input, None, None, None

# Usage helper
def leaky_clamp(input, min_val, max_val, leak=0.01):
    return LeakyClamp.apply(input, min_val, max_val, leak)
