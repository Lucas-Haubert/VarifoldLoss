# FlexForecast

Flexible Deep Learning Time Series Forecasting framework.

At this time, I have reproduced the experiments of the iTransformer repository (https://github.com/thuml/iTransformer/tree/main) from the server Ruche, I launched one baby experiment (1 epoch, run.py). 

I also extended the evaluation metrics files, in order to integrate normalized versions and DTW.

Next step: Focus on the losses. I must be able to train the network with MSE or MAE (ie L2 or L1: easy to get from torch.nn), but also with losses based on shape comparisons (ie DILATE or TILDE-Q).
