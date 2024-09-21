# The Varifold Loss: A Geometrical Approach to Loss Functions for Deep Learning Time Series Forecasting


## Overview

This repository is the official PyTorch implementation of the Varifold Loss, a shape-based loss function for training deep neural networks to perform time series forecasting. It aims at providing an easy-to-use pipeline where datasets, models, or loss functions can be easily swapped, with a particular emphasis on the latter.

This work is the result of my Master's Thesis for the MVA Master's program at the École Normale Supérieure Paris-Saclay, within the Signal Processing team at the [Centre Borelli](https://centreborelli.ens-paris-saclay.fr/en).


## The Varifold Loss

Often neglected, the design of the loss function is a crucial step in training a deep neural network for time series forecasting. The Mean Squarred Error (MSE) is the most widely used optimization metric, by convention. Yet, this approach is limiting when attempting to effectively capture the information from the signal in terms of shape and frequency fidelity. In this work, I give the construction of the so-called Varifold Loss which paves the way for a new paradigm in the design of loss functions based on shape spaces and kernel methods. 


## Usage

1. Install PyTorch and the necessary dependencies.

   ```bash
   pip install -r requirements.txt

2. Download the data from [Google Drive](https://drive.google.com/drive/folders/1OPz3pVgydOBUcxl9U0tVTTiScj12IQNc?usp=drive_link) and store them in the folder `./dataset`.

3. Train and evaluate the models with the different loss functions. The following scripts are examples of commands to reproduce the experiments of the thesis.

   ```
   # Noise sensitivity without trend
   bash ./scripts/Simple_SNR/MLP.sh

   # Noise sensitivity with trend
   bash ./scripts/Trend_SNR/DLinear.sh

   # Heatmap
   bash ./scripts/Fractal_2/MLP.sh

   # Multi-scale strategy
   bash ./scripts/Multi_Scale/DLinear.sh

   # Real-world univariate
   bash ./scripts/real_world_univariate/ETTh1/DLinear.sh

   # Real-world multivariate
   bash ./scripts/real_world_multivariate/ETTh1/DLinear.sh
   ```

## Sources

The code for the models (Autoformer, DLinear, SegRNN, TimesNet), as well as the structure of the repository, is inspired by the "Time Series Library":

```
@article{wang2024tssurvey,
  title={Deep Time Series Models: A Comprehensive Survey and Benchmark},
  author={Yuxuan Wang and Haixu Wu and Jiaxiang Dong and Yong Liu and Mingsheng Long and Jianmin Wang},
  booktitle={arXiv preprint arXiv:2407.13278},
  year={2024},
}
```

The first alternative loss function to MSE, DILATE, was developed by V. Le Guen and N. Thome:

```
@article{le2019shape,
  title={Shape and time distortion loss for training deep time series forecasting models},
  author={Le Guen, Vincent and Thome, Nicolas},
  journal={Advances in neural information processing systems},
  volume={32},
  year={2019}
}
```

The mathematical construction of oriented varifolds as a tool for curve comparison was introduced by I. Kaltenmark, B. Charlier, and N. Charon:

```
@inproceedings{kaltenmark2017general,
  title={A general framework for curve and surface comparison and registration with oriented varifolds},
  author={Kaltenmark, Irene and Charlier, Benjamin and Charon, Nicolas},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3346--3355},
  year={2017}
}
```


## Contact

If you have any questions or want to use the code, feel free to contact:

- Lucas Haubert ([lucas.haubert@ens-paris-saclay.fr](lucas.haubert@ens-paris-saclay.fr))


## Acknowledgement

I would like to acknowledge [Laurent Oudre](http://www.laurentoudre.fr/) for being my tutor during my research internship at the [Centre Borelli](https://centreborelli.ens-paris-saclay.fr/en).

I also thank my two supervisors [Chrysoula Kosma](https://www.linkedin.com/in/chrykosma/) and [Thibaut Germain](https://www.linkedin.com/in/thibaut-germain/) for helping me a lot during this period.
