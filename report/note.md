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
4. use adam optimizer,with default learning rate,Because I think low learning rate works best.
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
With different test it seems like low learning rate is doing best for HuberLoss.
![](./current_best.png)

## Experiment 4: 
Icreased the layer count 
(256, 128, 64,128,64 ,32) seeinf the feedback.As before,it plataued faster.
![](./more_layer.png)

## Experiment 5:
I changed the error to MSE and with increased layer mSE worked somehwat better,than HuberLoss(0.5),with scale_grad
regularization=0.1,also I reduced the batch size on this one,I don't know if huge has changed.
![](./mse_more_layer.png)

Still it wasn't better,Let's remove the scale_grad and see the result.
## Experiment 6: 
Ok having 
1. grad_scalar=True,
2. higher layer (256, 128, 64,128,64 ,32),
3. and higher regularization (0.5,0.5,0.5) 
4. MSE error,
5.
I have made the model learn much better.
Given I have kept the default optimizer settigns,
result:As it turns out the this approach has a lot of plataued values. Some time the result is not good as expected.
![](./mse1.png)
![](./mse2.png)
![](./mse3.png)

## Experiment 7:
Testing if changing the grad_scalar has positive or negative effect.It turns out current grad_scalar has good effect on the optimization given the regularization is high enough,
Current test similar to experiment 6,but with no grad scalar,Making.
![](./no_scale1.png)
![](./no_scale2.png)
![](./no_scale3.png)

## Experiment 8:
Because I previously find out grad_scalar really works great,I am gonna keep it in for the rest of the project,
Current test was doing if is comparred to regularization 2 if regularization 1 is decreased,So say (0.4,0.5),Will the training go bad,My previous research said it will,Now I am checking anew.Got better result.

![](./small_reg_1.png)
![](./small_reg_2.png)

It seems that regularization1 can't never be less than regularization2,I need to test what happens on same or higher.
## Experiment 9:

I keeped the reg1=0.4 and decreased reg2=0.4 so like to prove their is no effect of reg1=0.5,
I am getting the good behiavour like before,Now I need to check if reg1>reg2 WIll their be more effect or not.
![](./balance1.png)
![](./balance2.png)
It came to the position it was before.,

## Experiment 10:
 
Wanna see if I increase reg_1,Will it get as much performance.
Remember I am keeping regularization 2 constant,
This test check what is the behiavour of reg1 if I keep reg 2 constant,Here reg2=0.5,

![](./r1_increase.png)
![](./r1_increase2.png)
The test always gives that having r1=r2 regularization is the way to go.

## Experiment 11:
Checked which regularization works the best,as it turns out reg=0.5 works the best,Like always.
![](./reg1_change.png)
![](./reg1_change1.png)
![](./reg1_change2.png)

## Experiment 12:
Checked the right initialization method,It turns out odd=0,even=1 baised is the way to good
![](./right_init.png)
![](./right_init2.png)

Now other than normal distribution,I changed it to constant,
And I got really bad performace in result.
Plot isn't working,Even with 100 epoch I got .4999 So like random choice.

But there might be a sweet spot,Like If I have odd_even=(0.5,0.5) normal distribution,

Epoch 0099 | loss = 0.469072 | error = 0.347975 | reg = 0.302760 | tau_0 = 13.096493 | tau_1 = 19.820488 | tau_2 = 
8.476036 | tau_3 = 9.336477 | tau_4 = 8.504196 | tau_5 = 7.956133
