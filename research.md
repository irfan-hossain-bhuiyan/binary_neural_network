# Introduction:
Two of the importtant things of neural network is learning the blackbox that the neural netowork learned,and running faster on inference,As because of neural scaling law,people are making larger model making higher hardware demand,I am trying to help in both of those field by proposing a neural etwork architecture,On some regularization,It can mapped to boolean expression.Making highly performant in inference time.Our neural network architecture is a $\or w_i \and (z_i \xor bias)$ where $bias$ and $w_i$ are both learnable parameter.
Related Paper:
- Deep Differentiable Logic Gate Networks — Petersen et al. (NeurIPS 2022) it also creates a probalilistic boolean logic,but unlike circuit ,it is trained to create gate,Not boolean logic.
- There is Binary Neural Network,XNor Neural Network that make them faster in inference by removing Multilication from the operation.
- I am thinking of doing this entirely through training to learn boolean algebra.So not using addition at all.

- There is tnorm and gate equivalent of the differential logic,
Need to check but the issue with this I think is that It will leverage linearity in the model,So anneling it will result
in catastrophic learning loss.
## Paper to introduce:
- Deep Differentiable Logic Gate Networks Based on Fuzzy Łukasiewicz T-norm (2025).
- Deep Differentiable Logic Gate Networks(2022)
- Light Differentiable Logic Gate Networks" (Yousefi et al., 2025)
- Binary Neural Network
