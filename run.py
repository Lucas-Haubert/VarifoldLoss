import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import random
import numpy as np
import time
import json
import os

# def compute_mean_median_std_metrics(metrics_list):
#     metrics_array = np.array(metrics_list)
#     mean_metrics = np.mean(metrics_array, axis=0)
#     median_metrics = np.median(metrics_array, axis=0)
#     std_metrics = np.std(metrics_array, axis=0)
#     return mean_metrics, median_metrics, std_metrics

def compute_mean_median_std_metrics(metrics_list):
    # Extraire les noms des métriques à partir du premier dictionnaire
    metric_names = metrics_list[0].keys()

    # Initialiser des listes pour chaque métrique
    aggregated_metrics = {metric: [] for metric in metric_names}

    # Remplir les listes avec les valeurs de chaque métrique pour toutes les itérations
    for metrics in metrics_list:
        for metric in metric_names:
            aggregated_metrics[metric].append(metrics[metric])

    # Calculer les statistiques pour chaque métrique
    mean_metrics = {metric: np.mean(aggregated_metrics[metric]) for metric in metric_names}
    median_metrics = {metric: np.median(aggregated_metrics[metric]) for metric in metric_names}
    std_metrics = {metric: np.std(aggregated_metrics[metric]) for metric in metric_names}

    return mean_metrics, median_metrics, std_metrics

if __name__ == '__main__':
    # fix_seed = 2023
    # random.seed(fix_seed)
    # torch.manual_seed(fix_seed)
    # np.random.seed(fix_seed)

    seeds = [2023, 2024, 2025, 2026, 2027]  # Different seeds for each iteration
    metrics_results = []
    metrics_on_vali_results = []
    heatmap_dict = {}

    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')

    # data loader
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
    #parser.add_argument('--checkpoints', type=str, default='./outputs/checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
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

    # optimization
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

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # DILATE
    parser.add_argument('--alpha_dilate', type=float, default=0.5, help='alpha in dilate loss')

    # TILDE-Q
    parser.add_argument('--alpha_tildeq', type=float, default=0.5)
    parser.add_argument('--gamma_tildeq', type=float, default=0.01)

    # VARIFOLD
    parser.add_argument('--or_kernel', type=str, default="Gaussian")

    parser.add_argument('--sigma_t_1', type=float, default=1)
    parser.add_argument('--sigma_t_2', type=float, default=1)
    parser.add_argument('--sigma_s_1', type=float, default=10)
    parser.add_argument('--sigma_s_2', type=float, default=10)

    parser.add_argument('--sigma_t_1_little', type=float, default=10)
    parser.add_argument('--sigma_t_2_little', type=float, default=10)
    parser.add_argument('--sigma_t_1_big', type=float, default=10)
    parser.add_argument('--sigma_t_2_big', type=float, default=10)

    parser.add_argument('--sigma_s_1_little', type=float, default=10)
    parser.add_argument('--sigma_s_2_little', type=float, default=10)
    parser.add_argument('--sigma_s_1_big', type=float, default=10)
    parser.add_argument('--sigma_s_2_big', type=float, default=10)

    parser.add_argument('--sigma_t_1_kernel_1', type=float, default=10)
    parser.add_argument('--sigma_t_1_kernel_2', type=float, default=10)
    parser.add_argument('--sigma_t_1_kernel_3', type=float, default=10)
    parser.add_argument('--sigma_s_1_kernel_1', type=float, default=10)
    parser.add_argument('--sigma_s_1_kernel_2', type=float, default=10)
    parser.add_argument('--sigma_s_1_kernel_3', type=float, default=10)

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

    parser.add_argument('--heatmaps_base_name', type=str, required=False, default='heatmaps',
                        help='Base name for heatmaps without sigma specifications')


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

    if args.exp_name == 'partial_train':
        Exp = Exp_Long_Term_Forecast_Partial
    else:
        Exp = Exp_Long_Term_Forecast


    if args.is_training:
        for ii in range(args.itr):

            seed = seeds[ii % len(seeds)]  
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            setting = 'evalmode_{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.evaluation_mode,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii)

            # exp = Exp(args)
            # exp.plot_batches_without_prediction(setting)

            exp = Exp(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            start_training = time.time()
            exp.train(setting)
            end_training = time.time()
            print("Training with {} epochs: {} seconds".format(args.train_epochs, end_training - start_training))

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            start_testing = time.time()

            if args.evaluation_mode == 'raw':
                test_metrics = exp.test(setting, test=1)
            elif args.evaluation_mode == 'structural':
                test_metrics = exp.test_structural(setting, test=1)

            end_testing = time.time()
            print("Testing: {} seconds".format(end_testing - start_testing))

            vali_metrics = exp.test_on_vali(setting, test=1)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()

            metrics_results.append(test_metrics)
            metrics_on_vali_results.append(vali_metrics)

        mean_metrics, median_metrics, std_metrics = compute_mean_median_std_metrics(metrics_results)

        mean_vali_metrics, median_vali_metrics, std_vali_metrics = compute_mean_median_std_metrics(metrics_on_vali_results)

        # list_metrics = ['MSE', 'MAE', 'DTW', 'rFFT_low', 'rFFT_mid', 'rFFT_high', 'rSE']
        list_metrics = ['MSE', 'MAE', 'DTW', 'TDI']

        mean_metrics_dict = {metric: mean_metrics[metric] for metric in list_metrics}
        median_metrics_dict = {metric: median_metrics[metric] for metric in list_metrics}
        std_metrics_dict = {metric: std_metrics[metric] for metric in list_metrics}

        print('Mean metrics:', mean_metrics_dict)
        print('Median metrics:', median_metrics_dict)
        print('Standard deviation of metrics:', std_metrics_dict)

        mean_vali_metrics_dict = {metric: mean_vali_metrics[metric] for metric in list_metrics}
        median_vali_metrics_dict = {metric: median_vali_metrics[metric] for metric in list_metrics}
        std_vali_metrics_dict = {metric: std_vali_metrics[metric] for metric in list_metrics}

        print('Mean metrics on validation dataset:', mean_vali_metrics_dict)
        print('Median metrics on validation dataset:', median_vali_metrics_dict)
        print('Standard deviation of metrics on validation dataset:', std_vali_metrics_dict)

    else:
        ii = 0
        setting = 'evalmode_{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.evaluation_mode,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy, ii)

        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))

        if args.evaluation_mode == 'raw':
            test_metrics = exp.test(setting, test=1)
        elif args.evaluation_mode == 'structural':
            test_metrics = exp.test_structural(setting, test=1)

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

        

        
