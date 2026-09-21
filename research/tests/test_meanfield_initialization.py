import torch
from research.meanfield_initialization import target_selected_probability, meanfield_gaussian_metadata, meanfield_gaussian_edge_init_

def test_target_probability_fixed_point():
    for m in (8, 64, 784):
        s=target_selected_probability(m)
        assert abs((1-s/2)**m-.5)<1e-12

def test_gaussian_initializer_is_seeded_and_fan_in_aware():
    a=torch.empty(64,8); b=torch.empty(64,8)
    meanfield_gaussian_edge_init_(a,8,4.,torch.Generator().manual_seed(3))
    meanfield_gaussian_edge_init_(b,8,4.,torch.Generator().manual_seed(3))
    assert torch.equal(a,b)
    assert meanfield_gaussian_metadata(8,4.)['mu'] != meanfield_gaussian_metadata(64,4.)['mu']

def test_gaussian_initializer_uses_normal_raw_values():
    x=torch.empty(4096,64)
    meanfield_gaussian_edge_init_(x,64,4.,torch.Generator().manual_seed(0))
    assert x.std() > 3.5
    assert not torch.all((x==0) | (x==1))
