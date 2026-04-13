- Smooth gradient don't work,It is proven
- L1Loss don't work.It is proven
- gradient decay don't happen much
- When I see the distribution of my training I see the a lot of weights is saturated in the left side,Not a lot of weights saturated in the right side,It is because of the regulation,
So I am thinking Do I need to regulate from both side,
I feel like it.
Because we have seen having w.relu() error yield better result

- I removed DecreaseLROnPlateu in scheduler,SO I think i need to bring it back,I am not sure how improve it make it.
- Increased cost in regularization always make the model better in avg.
- Always use lower learning rate,Default ADAm settings is giving better result than scaled one
- Lower reg is always bad on avg,Not sure why
- Will try increasing as much as possible to see the effect.

**All of this test is happening 1 time,So it is not confident the result are authentic,Maybe some random make the test better.**

Newly Initialized model.
## Experiment 1:
1. Currently I have initialization for having mean =1 for even layer,and mean =0 for odd layer,I am now testing how much regularization helps in here.
2. had MSE error
3. I had a batch size of 256,Higher than 128
4. use adam optimizer,with default learning rate,Because I think low learning rate 
3. Their is a balance on what regulairzation works better.Here 0.05 works the best.
![](./regularization_test.png)
4. So I think using differnet loss function will influence the result differently

Newly initialized model
## Experiment 2: 
1. I aam gonna do the entire experiment on experiment 1,But with HuberLoss,I think this will work better.
2. batch size 128,Making some more rng.
3. Test which regularization works best.
4. Doing 100 epoch of training.
5. In simple glance,the HuberLoss is doing ridiculiously good,compared to MSELoss,It can because of both have different initialization.Or because it have more randomization because of 128 batch size.
HuberLoss(delta=0.1) results
![](./regularization_huberloss.png)

HuberLoss(delta=0.5) is the way to go.Didn't increase learning rate.
![](./huber_loss.png)
I increased the dimension to see how the performance get better.Previously I had  (256, 128, 64,32) and now I have 
(256, 128, 64,128,64 ,32),
![](./higher_dim.png)

## Experiment 3:
1. In HuberLoss(delta=0.5) 0.05 regularization ,I previously used lr=0.1 and betas:(0.5,0.5),I think what will work better here.
As the model is learning steadly.But also come the discretization part,I need to see what I can do?
