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


## Experiment 1:
Currently I have initialization for having mean =1 for even layer,and mean =0 for odd layer,I am now testing how much regularization helps in here.
