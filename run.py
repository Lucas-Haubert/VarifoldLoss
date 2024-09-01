import argparse
import torch
import random
import numpy as np
import time
import json
import os

from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.metrics import compute_mean_median_std_metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Internship')

    # Base configuration
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--script_name', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [MLP, LSTM, CNN, Transformer, iTransformer, SegRNN, TimesNet]')

    # Data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/synthetic/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='SNR_infty.csv', help='data csv file')
    parser.add_argument('--structural_data_path', type=str, default='SNR_infty.csv', help='data csv file')
    parser.add_argument('--evaluation_mode', type=str, default='raw', help='raw or structural')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./new_outputs/checkpoints/', help='location of model checkpoints')

    # Observation and horizon
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # Model hyperparameters
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # Optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--plot_metrics_and_gains', type=int, default=0, help='1 to plot metrics and gains curves over epochs')
    parser.add_argument('--bool_plot_examples_without_forecasts', type=int, default=0, help='1 to plot examples of time series in the dataset')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # TCN
    parser.add_argument('--out_dim_first_layer', type=int, default=32, help='then multiplied by 2 each layer')
    parser.add_argument('--fixed_kernel_size_tcn', type=int, default=2, help='fixed kernel size')

    # TimesNet
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')

    # SegRNN
    parser.add_argument('--rnn_type', default='gru', help='rnn_type')
    parser.add_argument('--dec_way', default='pmf', help='decode way')
    parser.add_argument('--seg_len', type=int, default=48, help='segment length')
    parser.add_argument('--win_len', type=int, default=48, help='windows length')
    parser.add_argument('--channel_id', type=int, default=1, help='Whether to enable channel position encoding')
    parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./dataset/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')

    # DILATE
    parser.add_argument('--alpha_dilate', type=float, default=0.5, help='alpha in dilate loss')

    # TILDE-Q
    parser.add_argument('--alpha_tildeq', type=float, default=0.5)
    parser.add_argument('--gamma_tildeq', type=float, default=0.01)

    # VARIFOLD

    # One kernel or two kernels in a sum
    parser.add_argument('--number_of_kernels', type=int, default=1, help='2 if sum of kernels)')

    # One kernel
    parser.add_argument('--position_kernel', type=str, default="Gaussian", help='Gaussian of Cauchy, for OneKernel')
    parser.add_argument('--sigma_t_pos', type=float, default=1)
    parser.add_argument('--sigma_s_pos', type=float, default=1)
    parser.add_argument('--orientation_kernel', type=str, default="Distribution", help='Distribution, Current, UnorientedVarifold or OrientedVarifold, for OneKernel')
    parser.add_argument('--sigma_t_or', type=float, default=1)
    parser.add_argument('--sigma_s_or', type=float, default=1)

    # Two kernels
    parser.add_argument('--position_kernel_little', type=str, default="Gaussian", help='Gaussian of Cauchy')
    parser.add_argument('--sigma_t_pos_little', type=float, default=1)
    parser.add_argument('--sigma_s_pos_little', type=float, default=1)

    parser.add_argument('--orientation_kernel_little', type=str, default="Distribution", help='Distribution, Current, UnorientedVarifold or OrientedVarifold')
    parser.add_argument('--sigma_t_or_little', type=float, default=1)
    parser.add_argument('--sigma_s_or_little', type=float, default=1)

    parser.add_argument('--position_kernel_big', type=str, default="Gaussian", help='Gaussian of Cauchy')
    parser.add_argument('--sigma_t_pos_big', type=float, default=1)
    parser.add_argument('--sigma_s_pos_big', type=float, default=1)

    parser.add_argument('--orientation_kernel_big', type=str, default="Distribution", help='Distribution, Current, UnorientedVarifold or OrientedVarifold')
    parser.add_argument('--sigma_t_or_big', type=float, default=1)
    parser.add_argument('--sigma_s_or_big', type=float, default=1)

    parser.add_argument('--weight_little', type=float, default=0.5)
    parser.add_argument('--weight_big', type=float, default=0.5)

    # Heatmaps  
    parser.add_argument('--heatmaps_base_name', type=str, required=False, default='heatmaps',
                        help='Base name for heatmaps without parameters specifications')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        print("We use a GPU")
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)
    
    Exp = Exp_Long_Term_Forecast

    learning_rate_str = str(args.learning_rate).replace('.', 'dot')

    if args.loss == 'MSE':
                setting = 'evalmode_{}_{}_{}_{}_ft{}_W{}_H{}_Loss{}_Epo{}_Pat{}_B{}_lr{}'.format(
                    args.evaluation_mode,
                    args.script_name,
                    args.model,
                    args.data_path.split('.')[0],
                    args.features,
                    args.seq_len,
                    args.pred_len,
                    args.loss,
                    args.train_epochs,
                    args.patience,
                    args.batch_size,
                    learning_rate_str)

    elif args.loss == 'DILATE':
        setting = 'evalmode_{}_{}_{}_{}_ft{}_W{}_H{}_Loss{}_Alpha{}_Epo{}_Pat{}_B{}_lr{}'.format(
            args.evaluation_mode,
            args.script_name,
            args.model,
            args.data_path.split('.')[0],
            args.features,
            args.seq_len,
            args.pred_len,
            args.loss,
            args.alpha_dilate,
            args.train_epochs,
            args.patience,
            args.batch_size,
            learning_rate_str)

    elif args.loss == 'TILDEQ':
        setting = 'evalmode_{}_{}_{}_{}_ft{}_W{}_H{}_Loss{}_Alpha{}_Gamma{}_Epo{}_Pat{}_B{}_lr{}'.format(
            args.evaluation_mode,
            args.script_name,
            args.model,
            args.data_path.split('.')[0],
            args.features,
            args.seq_len,
            args.pred_len,
            args.loss,
            args.alpha_tildeq,
            args.gamma_tildeq,
            args.train_epochs,
            args.patience,
            args.batch_size,
            learning_rate_str)

    elif args.loss == 'VARIFOLD':
        if args.number_of_kernels == 1:
            setting = '{}_{}_{}_{}_ft{}_W{}_H{}_Loss{}_PKer{}_{}_{}_OKer{}_{}_{}'.format(
                args.evaluation_mode,
                args.script_name,
                args.model,
                args.data_path.split('.')[0],
                args.features,
                args.seq_len,
                args.pred_len,
                args.loss,
                args.position_kernel,
                args.sigma_t_pos,
                args.sigma_s_pos,
                args.orientation_kernel,
                args.sigma_t_or,
                args.sigma_s_or)
        elif args.number_of_kernels == 2:
            setting = '{}_{}_{}_{}_ft{}_W{}_H{}_Loss{}_Pl{}_{}_{}_Ol{}_{}_{}_Pb{}_{}_{}_Ob{}_{}_{}_w_l{}_w_b{}'.format(
                args.evaluation_mode,
                args.script_name,
                args.model,
                args.data_path.split('.')[0],
                args.features,
                args.seq_len,
                args.pred_len,
                args.loss,
                args.position_kernel_little,
                args.sigma_t_pos_little,
                args.sigma_s_pos_little,
                args.orientation_kernel_little,
                args.sigma_t_or_little,
                args.sigma_s_or_little,
                args.position_kernel_big,
                args.sigma_t_pos_big,
                args.sigma_s_pos_big,
                args.orientation_kernel_big,
                args.sigma_t_or_big,
                args.sigma_s_or_big,
                args.weight_little,
                args.weight_big)

    metrics_results = []
    metrics_on_vali_results = []

    seeds = [2024 + i for i in range(args.itr)]

    if args.is_training:
        for ii in range(args.itr):

            seed = seeds[ii]  
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            setting_iter = setting + '_Iteration{}'.format(ii)

            exp = Exp(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting_iter))
            start_training = time.time()
            exp.train(setting_iter)
            end_training = time.time()
            print("Training with {} epochs: {} seconds".format(args.train_epochs, end_training - start_training))

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting_iter))
            start_testing = time.time()

            if args.evaluation_mode == 'raw':
                test_metrics = exp.test(setting_iter, test=1)
            elif args.evaluation_mode == 'structural':
                test_metrics = exp.test_structural(setting_iter, test=1)

            end_testing = time.time()
            print("Testing: {} seconds".format(end_testing - start_testing))

            vali_metrics = exp.test_on_vali(setting_iter, test=1)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting_iter))
                exp.predict(setting_iter, True)

            torch.cuda.empty_cache()

            metrics_results.append(test_metrics)
            metrics_on_vali_results.append(vali_metrics)

        mean_metrics, median_metrics, std_metrics = compute_mean_median_std_metrics(metrics_results)
        mean_vali_metrics, median_vali_metrics, std_vali_metrics = compute_mean_median_std_metrics(metrics_on_vali_results)

        formated_mean_metrics = ", ".join([f"{metric}:{mean_metrics[metric]}" for metric in mean_metrics.keys()])
        formated_median_metrics = ", ".join([f"{metric}:{median_metrics[metric]}" for metric in median_metrics.keys()])
        formated_std_metrics = ", ".join([f"{metric}:{std_metrics[metric]}" for metric in std_metrics.keys()])
        formated_mean_vali_metrics = ", ".join([f"{metric}:{mean_vali_metrics[metric]}" for metric in mean_vali_metrics.keys()])
        formated_median_vali_metrics = ", ".join([f"{metric}:{median_vali_metrics[metric]}" for metric in median_vali_metrics.keys()])
        formated_std_vali_metrics = ", ".join([f"{metric}:{std_vali_metrics[metric]}" for metric in std_vali_metrics.keys()])

        folder_path = './new_outputs/numerical_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        file_path = os.path.join(folder_path, f"txt_metrics_global_{setting}.txt")
        with open(file_path, 'a') as f:
            f.write(setting + "  \n")
            f.write('Number of iterations : ' + str(args.itr) + '\n')
            f.write('Mean metrics :' + formated_mean_metrics + '\n')
            f.write('Median metrics :' + formated_median_metrics + '\n')
            f.write('Standard deviation of metrics :' + formated_std_metrics + '\n')
            f.write('Mean metrics on validation dataset :' + formated_mean_vali_metrics + '\n')
            f.write('Median metrics on validation dataset :' + formated_median_vali_metrics + '\n')
            f.write('Standard deviation of metrics on validation dataset :' + formated_std_vali_metrics + '\n')
            f.write('\n')
            f.write('\n')

        print('Mean metrics:', mean_metrics)
        print('Median metrics:', median_metrics)
        print('Standard deviation of metrics:', std_metrics)

        print('Mean metrics on validation dataset:', mean_vali_metrics)
        print('Median metrics on validation dataset:', median_vali_metrics)
        print('Standard deviation of metrics on validation dataset:', std_vali_metrics)






    # Mettre à part

    # Revoir cette partie pour fonction de base de is_training == 0, mais aussi calcul heatmap dessus
    
    else:
        heatmap_dict = {}

        ii = 0

        setting_iter = setting + '_Iteration{}'.format(ii)

        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting_iter))

        if args.evaluation_mode == 'raw':
            test_metrics = exp.test(setting_iter, test=1)
        elif args.evaluation_mode == 'structural':
            test_metrics = exp.test_structural(setting_iter, test=1)

        end_testing = time.time()
        print("Testing: {} seconds".format(end_testing - start_testing))

        vali_metrics = exp.test_on_vali(setting_iter, test=1)

        torch.cuda.empty_cache()

        mean_metrics, median_metrics, std_metrics = compute_mean_median_std_metrics([metrics_results])
        mean_vali_metrics, median_vali_metrics, std_vali_metrics = compute_mean_median_std_metrics([metrics_on_vali_results])

        formated_mean_metrics = ", ".join([f"{metric}:{mean_metrics[metric]}" for metric in mean_metrics.keys()])
        formated_median_metrics = ", ".join([f"{metric}:{median_metrics[metric]}" for metric in median_metrics.keys()])
        formated_std_metrics = ", ".join([f"{metric}:{std_metrics[metric]}" for metric in std_metrics.keys()])
        formated_mean_vali_metrics = ", ".join([f"{metric}:{mean_vali_metrics[metric]}" for metric in mean_vali_metrics.keys()])
        formated_median_vali_metrics = ", ".join([f"{metric}:{median_vali_metrics[metric]}" for metric in median_vali_metrics.keys()])
        formated_std_vali_metrics = ", ".join([f"{metric}:{std_vali_metrics[metric]}" for metric in std_vali_metrics.keys()])

        folder_path = './new_outputs/numerical_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        file_path = os.path.join(folder_path, f"txt_metrics_global_{setting}.txt")
        with open(file_path, 'a') as f:
            f.write(setting + "  \n")
            f.write('Number of iterations : ' + str(args.itr) + '\n')
            f.write('Mean metrics :' + formated_mean_metrics + '\n')
            f.write('Median metrics :' + formated_median_metrics + '\n')
            f.write('Standard deviation of metrics :' + formated_std_metrics + '\n')
            f.write('Mean metrics on validation dataset :' + formated_mean_vali_metrics + '\n')
            f.write('Median metrics on validation dataset :' + formated_median_vali_metrics + '\n')
            f.write('Standard deviation of metrics on validation dataset :' + formated_std_vali_metrics + '\n')
            f.write('\n')
            f.write('\n')

        print('Mean metrics:', mean_metrics)
        print('Median metrics:', median_metrics)
        print('Standard deviation of metrics:', std_metrics)

        print('Mean metrics on validation dataset:', mean_vali_metrics)
        print('Median metrics on validation dataset:', median_vali_metrics)
        print('Standard deviation of metrics on validation dataset:', std_vali_metrics)


        """

        heatmap_file_path = os.path.join('new_outputs', 'numerical_results', f'{args.heatmaps_base_name}.json')

        try:
            if os.path.exists(heatmap_file_path):
                with open(heatmap_file_path, 'r') as json_file:
                    heatmap_dict = json.load(json_file)
            else:
                heatmap_dict = {}
        except json.JSONDecodeError as e:
            print(f"Erreur lors du chargement du fichier JSON : {e}")
            heatmap_dict = {}

        for metric_name, metric_value in test_metrics.items():
            if metric_name not in heatmap_dict:
                heatmap_dict[metric_name] = {}
            sigma_t_1_str = str(args.sigma_t_1).replace('.', 'dot')
            sigma_s_1_str = str(args.sigma_s_1).replace('.', 'dot')
            heatmap_dict[metric_name][(sigma_t_1_str, sigma_s_1_str)] = metric_value

        with open(heatmap_file_path, 'w') as json_file:
            json.dump(heatmap_dict, json_file, indent=4)

        torch.cuda.empty_cache()

        """
