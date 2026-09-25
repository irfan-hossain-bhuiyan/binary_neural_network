import torch
from research.run_a5_target_gini import LAMBDA, loss_parts, scalar_checks

def test_gini_scalar_endpoints_and_midpoint():
    p=torch.tensor([0., .5, 1.])
    for y, expected in [(0.,0),(1.,2)]:
        out=torch.tensor([y],requires_grad=True)
        mse,g=loss_parts(out,torch.tensor([y]))
        assert torch.isfinite(mse+LAMBDA*g)
    checks=scalar_checks()
    assert checks['0']['gini_argmin']==0.0
    assert checks['1']['gini_argmin']==1.0
    assert checks['q_0.5']['mid_is_max']
    assert checks['q_0.5']['mid_risk'] > checks['q_0.5']['endpoint_risk']

def test_gini_term_is_unnormalized_variance():
    p=torch.tensor([[.2,.8]])
    y=torch.tensor([[0.,1.]])
    mse,g=loss_parts(p,y)
    assert torch.allclose(g, torch.tensor(.16))
    assert torch.allclose(mse, torch.tensor(.04))
