import torch

from discrete_logic_net import DiscreteModernLogicGateNet
from research.analyze_meanfield_i1r import discrete_residual_trace


def test_i1r_discrete_residual_trace_uses_exact_xor_stages():
    model = DiscreteModernLogicGateNet(2, 1, width=2, num_residual_blocks=1)
    with torch.no_grad():
        model.stem.weight.zero_(); model.stem.bias.zero_()
        model.blocks[0].layer1.weight.zero_(); model.blocks[0].layer1.bias.zero_()
        model.blocks[0].layer2.weight.zero_(); model.blocks[0].layer2.bias.zero_()
        model.head.weight.zero_(); model.head.bias.zero_()
        model.stem.weight[0, 0] = True
        model.blocks[0].layer1.weight[0, 0] = True
        model.blocks[0].layer2.weight[0, 0] = True
        model.head.weight[0, 0] = True
    x = torch.tensor([[False, False], [False, True], [True, False], [True, True]])
    trace = dict(discrete_residual_trace(model, x))
    # The residual stage is a Boolean XOR of the block input and branch.
    assert torch.equal(trace['stem'], model.stem(x))
    assert torch.equal(trace['block0.layer1'], model.blocks[0].layer1(trace['stem']))
    assert torch.equal(trace['block0.layer2'], model.blocks[0].layer2(trace['block0.layer1']))
    assert torch.equal(trace['block0.residual'], trace['stem'] ^ trace['block0.layer2'])
    assert torch.equal(trace['head'], model.head(trace['block0.residual']))
    assert all(t.dtype == torch.bool for t in trace.values())
