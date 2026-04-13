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

## Methodology
Nor gate is enough to make any logical circuit,
Every neuron is is basically depends on the neuron of previous layer
$$\or (w_i \cap x_i)\xor b$$ where wi and b is the learnable parameter of the model,$w_i$ and $b_i$ is constraint to 0 or 1.
Like neural network given enough depth and width we can mimic any logical relation.

We will map the gate in a continious domain
$x_i \and w_i=x_i*w_i$ and or gate will be softmax(z*tau)*z to get the expected max,$tau=1/temprature$ and tau is encouraged to grow.
and b is beta make if the or gate should passed with not gate.
## Training 1
I will start with training xor gate.So 16 bit +16 bit=32 bit input,The setup is 
1. 100000 sample
2. Layers =(256, 128, 64, 32)
3. No regularization has been added.
4. The bais is initialized to 1
5. No lr scheduler
6. adam optimizer with {"betas":(0.25,0.25),"lr":0.1},

## Observation:
1. On basic observation,higher regularization result in better performance,Need to plot graph of all of them
