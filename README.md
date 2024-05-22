# FlexForecast

Flexible Deep Learning Time Series Forecasting framework.

NB: This README file is more a rough draft than a real README. Now it stands to present my work to the supervisors. It will be reshaped soon. 

At this time, I have reproduced the experiments of the iTransformer repository (https://github.com/thuml/iTransformer/tree/main) from the server Ruche, I launched one baby experiment (1 epoch, run.py). 

I also extended the evaluation metrics files, in order to integrate normalized versions and DTW.

Next step: Focus on the losses. I have adapted the criterion function in order to train the network with either MSE or MAE. The goal is now to plug DILATE and TILDE-Q losses in the procedure, so that I can switch easily.
