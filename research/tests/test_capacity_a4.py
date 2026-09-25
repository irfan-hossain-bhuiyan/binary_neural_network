from research.capacity_a4 import make_capacity_net
from research.boolean_tasks import build_binary_addition


def test_current_discrete_architecture_represents_four_bit_addition():
    task = build_binary_addition({"bits": 4})
    net = make_capacity_net()
    got = net(task["X"].bool())
    expected = task["Y"].bool()
    assert bool((got == expected).all())
    assert bool((got == expected).all(dim=1).all())
