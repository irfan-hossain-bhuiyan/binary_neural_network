import torch
from research.meanfield_initialization import target_selected_probability, meanfield_gaussian_metadata, meanfield_gaussian_edge_init_, gaussian_bias_init_

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

def test_sigma_pairing_preserves_thresholded_edge_mask():
    masks=[]; values=[]
    for sigma in (2.,4.,6.):
        x=torch.empty(64,64)
        meanfield_gaussian_edge_init_(x,64,sigma,torch.Generator().manual_seed(77))
        values.append(x.clone()); masks.append(x>=0)
    assert torch.equal(masks[0], masks[1])
    assert torch.equal(masks[0], masks[2])
    assert not torch.equal(values[0], values[1])

def test_paired_biases_preserve_polarity_but_change_continuous_sensitivity():
    current=torch.empty(64,64); polarized=torch.empty(64,64)
    zseed=91
    gaussian_bias_init_(current,.5,.1,torch.Generator().manual_seed(zseed))
    gaussian_bias_init_(polarized,.5,2.,torch.Generator().manual_seed(zseed))
    current=current.clamp(0,1); polarized=polarized.clamp(0,1)
    assert torch.equal(current>=.5, polarized>=.5)
    gain_current=(1-2*current).abs().mean()
    gain_polarized=(1-2*polarized).abs().mean()
    assert gain_polarized > gain_current * 3
