import inspect
from typing import Any, Callable, Dict, Tuple
from rich.console import Console
from core_types import LayerGradStats

CONSOLE = Console()

def _call_matching(func: Callable, arg_dict: Dict[str, Any]) -> Any:
    """Call func forwarding only the kwargs its signature accepts."""
    valid = set(inspect.signature(func).parameters)
    return func(**{k: v for k, v in arg_dict.items() if k in valid})


def _format_peek(peek_results: Dict[str, Any]) -> str:
    return " | ".join(
        f"{k} = {v:.6f}" if isinstance(v, float) else f"{k} = {v}"
        for k, v in peek_results.items()
    )


def _accumulate_grad_stats(
    acc: Dict[str, LayerGradStats],
    new: Dict[str, LayerGradStats],
) -> Dict[str, LayerGradStats]:
    """Sum two LayerGradStats dicts element-wise (for later averaging)."""
    result = dict(acc)
    for name, s in new.items():
        if name in result:
            prev = result[name]
            result[name] = LayerGradStats(
                mean_abs        = prev.mean_abs        + s.mean_abs,
                norm_normalized = prev.norm_normalized + s.norm_normalized,
                max_abs         = prev.max_abs         + s.max_abs,
            )
        else:
            result[name] = s
    return result


def _divide_grad_stats(acc: Dict[str, LayerGradStats], n: int) -> Dict[str, LayerGradStats]:
    return {
        name: LayerGradStats(
            mean_abs        = (s.mean_abs        / n).detach().cpu(),
            norm_normalized = (s.norm_normalized / n).detach().cpu(),
            max_abs         = (s.max_abs         / n).detach().cpu(),
        )
        for name, s in acc.items()
    }
