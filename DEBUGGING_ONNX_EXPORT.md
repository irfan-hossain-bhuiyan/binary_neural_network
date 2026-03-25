# Debugging PyTorch ONNX Export in Binary Transformer

This document summarizes the debugging and resolution process for the `torch.onnx.export` errors encountered while exporting the custom binary transformer model.

## Issue 1: `torch.clamp` Type Promotion Error (FX Decomposition)

**Error:**
```python
AssertionError: Expected same OpOverload packet, got prim.device != aten.clamp
```

**Root Cause:**
In `prelude.py`, the `LeakyClamp` forward pass called `torch.clamp(input, min_val, max_val)` where `min_val` and `max_val` were treated as scalar tensors. When exporting to ONNX using the newer Dynamo backend (`torch.export`), the FX decomposition and type-promotion passes failed because it expected primitive numeric values (floats) for the clamp limits rather than scalar tensors.

**Fix:**
Converted the `min_val` and `max_val` arguments to strict Python floats to ensure standard type promotion.
```python
# Before
return torch.clamp(input, min_val, max_val)

# After
return torch.clamp(input, min=float(min_val), max=float(max_val))
```

## Issue 2: Moving from `dynamic_axes` to `dynamic_shapes`

**Warning:**
```
UserWarning: 'dynamic_axes' is not recommended when dynamo=True... Supply the 'dynamic_shapes' argument instead if export is unsuccessful.
```

**Root Cause:**
The original ONNX export call used the deprecated `dynamic_axes` argument, which doesn't map directly to the modern PyTorch 2.x `torch.export` (Dynamo) pipeline.

**Fix:**
Replaced `dynamic_axes` with the modern `dynamic_shapes` format. Also added `self.model.eval()` right before the export to suppress warnings about the model being in training mode, which ensures inference configuration is preserved.

## Issue 3: `dynamic_shapes` Argument Mapping Error

**Error:**
```python
torch._dynamo.exc.UserError: When `dynamic_shapes` is specified as a dict, its top-level keys must be the arg names ['x'] of `inputs`, but here they are ['input', 'output'].
```

**Root Cause:**
Initially, I translated `dynamic_axes` directly into `dynamic_shapes={"input": ..., "output": ...}`. However, `dynamic_shapes` behaves differently:
1. Keys must exactly match the parameter names in the model's `forward` function definition. Here, the model expects an argument named `x`, not `input`.
2. It only defines the shapes for *inputs*. Outputs are automatically inferred by PyTorch.

**Fix:**
Updated the `dynamic_shapes` dictionary to use the correct argument name `x` and removed the `output` entry entirely.
```python
# Before
dynamic_shapes={"input": {0: torch.export.Dim("batch_size", min=1)}, "output": {0: torch.export.Dim("batch_size", min=1)}}

# After
dynamic_shapes={"x": {0: torch.export.Dim("batch_size", min=1)}}
```

## Result
Running the export script (`python binray_transformer.py`) now successfully completes the FX tracing, decomposition, and translation steps, cleanly outputting `model.onnx` without any errors.