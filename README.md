# FlexForecast

Flexible Deep Learning Time Series Forecasting framework.

At this time, I have reproduced the experiments of the iTransformer repository (https://github.com/thuml/iTransformer/tree/main) from the server Ruche, I launched one baby experiment (1 epoch, run.py). 

I also extended the evaluation metrics files, in order to integrate normalized versions and DTW.

Next step: Focus on the losses. I must be able to train the network with MSE or MAE (ie L2 or L1: easy to get from torch.nn), but also with losses based on shape comparisons (ie DILATE or TILDE-Q). In a first time, I will then adapt the code to be able to switch easily between L1 and L2, then I focus on DILATE / TILDE-Q.

NB: My implementation of DTW seems quite naive (very long execution to compute the results of DTW => explore a clever / already implemented solution ; the solution may also be in the dilate or tilde q repo, since they involve DTW).
