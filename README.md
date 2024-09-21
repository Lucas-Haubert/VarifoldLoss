# The Varifold Loss: A Geometrical Approach to Loss Functions for Deep Learning Time Series Forecasting


## Overview

This repository is the official PyTorch implementation of the Varifold Loss, a shape-based loss function for training deep neural networks to perform time series forecasting. It aims at providing an easy-to-use pipeline where datasets, models, or loss functions can be easily swapped, with a particular emphasis on the latter.

This work is the result of my Master's Thesis for the MVA Master's program at the École Normale Supérieure Paris-Saclay, within the Signal Processing team at the [Centre Borelli](https://centreborelli.ens-paris-saclay.fr/en).


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

## Source

The code for the models (Autoformer, DLinear, SegRNN, TimesNet), as well as the structure of the repository, is inspired from the "Times Series Library":

```
@article{wang2024tssurvey,
  title={Deep Time Series Models: A Comprehensive Survey and Benchmark},
  author={Yuxuan Wang and Haixu Wu and Jiaxiang Dong and Yong Liu and Mingsheng Long and Jianmin Wang},
  booktitle={arXiv preprint arXiv:2407.13278},
  year={2024},
}
```


## Contact

If you have any questions or want to use the code, feel free to contact:

- Lucas Haubert ([lucas.haubert@ens-paris-saclay.fr](lucas.haubert@ens-paris-saclay.fr))


## Acknowledgement

I would like to acknowledge [Laurent Oudre](http://www.laurentoudre.fr/) for being my tutor during my research internship at the [Centre Borelli](https://centreborelli.ens-paris-saclay.fr/en).

I also thank my two supervisors [Chrysoula Kosma](https://www.linkedin.com/in/chrykosma/) and [Thibaut Germain](https://www.linkedin.com/in/thibaut-germain/) for helping me a lot during this period.
