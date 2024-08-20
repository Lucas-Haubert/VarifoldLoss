import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

from data_provider.data_factory import data_provider, data_provider_structural
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import compute_metrics

from loss.dilate.dilate_loss import DILATE
from loss.tildeq import tildeq_loss
from loss.varifold import VarifoldLoss, OneKernel, TwoKernels

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

        self.train_losses = []
        self.vali_losses = []
        self.test_losses = []

        self.name_of_dataset = self.args.data_path

        self.list_of_metrics = ['MSE', 'MAE', 'DTW', 'TDI']
        self.metrics_for_plots_over_epochs = {metric: {'val': [], 'test': []} for metric in self.list_of_metrics}

        self.gains_test_loss = []
        self.gains_test_metrics = {metric: [] for metric in self.list_of_metrics}

        self.number_of_actual_epochs = 0

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _get_data_structural(self, flag):
        data_set, data_loader = data_provider_structural(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'MSE':
            criterion = nn.MSELoss()
        elif self.args.loss == 'DILATE':
            criterion = lambda x,y: DILATE(x,y, alpha = self.args.alpha_dilate)
        elif self.args.loss == "TILDEQ":
            criterion = lambda x,y: tildeq_loss(x,y, alpha = self.args.alpha_tildeq)
        elif self.args.loss == "VARIFOLD":
            if self.args.number_of_kernels == 1:
                K = OneKernel(position_kernel = self.args.position_kernel, orientation_kernel = self.args.orientation_kernel, 
                              sigma_t_pos = self.args.sigma_t_pos, sigma_s_pos = self.args.sigma_s_pos, 
                              sigma_t_or = self.args.sigma_t_or, sigma_s_or = self.args.sigma_s_or, 
                              n_dim = self.args.enc_in + 1, device=self.device)
            elif self.args.number_of_kernels == 2:
                K = TwoKernels(position_kernel_little = self.args.position_kernel_little, orientation_kernel_little = self.args.orientation_kernel_little, 
                               position_kernel_big = self.args.position_kernel_big, orientation_kernel_big = self.args.orientation_kernel_big, 
                               sigma_t_pos_little = self.args.sigma_t_pos_little, sigma_s_pos_little = self.args.sigma_s_pos_little, 
                               sigma_t_or_little = self.args.sigma_t_or_little, sigma_s_or_little = self.args.sigma_s_or_little, 
                               sigma_t_pos_big = self.args.sigma_t_pos_big, sigma_s_pos_big = self.args.sigma_s_pos_big, 
                               sigma_t_or_big = self.args.sigma_t_or_big, sigma_s_or_big = self.args.sigma_s_or_big, 
                               weight_little = self.args.weight_little, weight_big = self.args.weight_big, 
                               n_dim = self.args.enc_in + 1, device=self.device)
        loss_function = VarifoldLoss(K, device=self.device)
        criterion = lambda x,y: loss_function(x,y)

        return criterion

    def cumulative_computing_loss_metrics(self, dataloader, criterion):
        total_loss = []

        preds = []
        trues = []

        metrics_over_batches = {'MSE': [], 'DTW': [], 'TDI': [], 'DILATE': [], 'VARIFOLD': []}

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataloader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)

                loss = loss.detach().cpu().numpy()

                total_loss.append(loss)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                pred = pred.numpy()
                true = true.numpy()

                metrics_current_batch = compute_metrics(pred, true, self.name_of_dataset)

                for metric in self.list_of_metrics:
                    metrics_over_batches[metric].append(metrics_current_batch[metric])
                
                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        metrics_current_epoch = compute_metrics(preds, trues, self.name_of_dataset) 

        total_loss = np.average(total_loss)
        
        return total_loss, metrics_current_epoch, metrics_over_batches

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def plot_losses_and_metrics(self, metric_name, setting):
        folder_path = './new_outputs/losses_and_metrics_wrt_epochs/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.plot(range(1, len(self.train_losses) + 1), self.train_losses, marker='.', color='blue', label='Training Loss')
        ax1.plot(range(1, len(self.vali_losses) + 1), self.vali_losses, marker='.', color='green', label='Validation Loss')
        ax1.plot(range(1, len(self.test_losses) + 1), self.test_losses, marker='.', color='red', label='Test Loss')
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Metric')
        validation_metric = self.metrics_for_plots_over_epochs[metric_name]['val']
        test_metric = self.metrics_for_plots_over_epochs[metric_name]['test']
        ax2.plot(range(1, len(validation_metric) + 1), validation_metric, marker='^', linestyle='--', color='black', label='Validation Metric')
        ax2.plot(range(1, len(test_metric) + 1), test_metric, marker='s', color='black', label='Test Metric') 
        ax2.tick_params(axis='y')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.title(f"Model: {self.args.model}, Loss: {self.args.loss}, Metric: {metric_name}, W: {self.args.seq_len}, H: {self.args.pred_len}, Dataset: {self.args.data_path}", pad=20)
        plt.savefig(os.path.join(folder_path, f"Loss_{self.args.loss}_Metric_{metric_name}_over_epochs.png"))

    def plot_gains_wrt_epochs(self, metric_name, setting):
        folder_path = './new_outputs/gains_wrt_epochs/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.set_xlabel('Epochs')
        ax.set_ylabel('Gain')
        ax.plot(range(1, len(self.gains_test_loss) + 1), self.gains_test_loss, marker='.', color='red', label='Test Loss Gain (%)')
        ax.plot(range(1, len(self.gains_test_metrics[metric_name]) + 1), self.gains_test_metrics[metric_name], marker='s', color='black', label='Test Metric Gain (%)')
        ax.tick_params(axis='y')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        ax.legend(loc='upper right')
        plt.title(f"Model: {self.args.model}, Loss: {self.args.loss}, Metric: {metric_name}, W: {self.args.seq_len}, H: {self.args.pred_len}, Dataset: {self.args.data_path}", pad=20)
        plt.savefig(os.path.join(folder_path, f"Gains_for_Loss_{self.args.loss}_Metric_{metric_name}_over_epochs.png"))

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        #path = os.path.join(self.args.checkpoints, setting)
        path = './new_outputs/checkpoints/' + setting
        if not os.path.exists(path):
            os.makedirs(path)

        
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            """print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            start_time_metrics = time.time()

            train_loss = np.average(train_loss)
            vali_loss, vali_metrics_current_epoch, vali_metrics_over_batches = self.cumulative_computing_loss_metrics(vali_loader, criterion)
            test_loss, test_metrics_current_epoch, test_metrics_over_batches = self.cumulative_computing_loss_metrics(test_loader, criterion)
            self.train_losses.append(train_loss)
            self.vali_losses.append(vali_loss)
            self.test_losses.append(test_loss)

            for metric in self.list_of_metrics:
                self.metrics_for_plots_over_epochs[metric]['val'].append(vali_metrics_current_epoch[metric])
                self.metrics_for_plots_over_epochs[metric]['test'].append(test_metrics_current_epoch[metric])

            # For plotting the gains on the test dataset
            if epoch == 0:
                initial_test_loss = test_loss
                initial_test_metrics = test_metrics_current_epoch  
            self.gains_test_loss.append(100*(initial_test_loss-test_loss)/initial_test_loss)
            for metric in self.list_of_metrics:
                    self.gains_test_metrics[metric].append(100*(initial_test_metrics[metric]-test_metrics_current_epoch[metric])/initial_test_metrics[metric])

            end_time_metrics = time.time()
            print("Time to compute metrics after epoch", epoch + 1, ":", end_time_metrics - start_time_metrics)"""

            # Just keep the essential (remove if i use cumulative function)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                self.number_of_actual_epochs = epoch + 1
                break

            """print('Validation and test MSE/DTW after epoch', epoch+1, ":")
            print('Validation MSE :', vali_metrics_current_epoch['MSE'])
            print('Test MSE :', test_metrics_current_epoch['MSE'])
            print('Validation DTW :', vali_metrics_current_epoch['DTW'])
            print('Test DTW :', test_metrics_current_epoch['DTW'])"""

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        
        """
        for metric in self.list_of_metrics:
            self.plot_losses_and_metrics(metric, setting)
            self.plot_gains_wrt_epochs(metric, setting)
        """
    
        return self.model

    def test(self, setting, test=0):
        print('Beginning the test')
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            #self.model.load_state_dict(torch.load(os.path.join('./outputs/checkpoints/' + setting, 'checkpoint.pth')))
            self.model.load_state_dict(torch.load(os.path.join('./new_outputs/checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        #folder_path = './outputs/visual_results/' + setting + '/'
        folder_path = './new_outputs/visual_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                number_samples = 3
                number_batches = len(test_loader)
                plot_indices = [i * number_batches // number_samples for i in range(number_samples)]
                
                if i in plot_indices:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

                    plt.figure(figsize=(10, 5))
                    plt.plot(gt, color='darkblue', label='Ground Truth')
                    plt.plot(pd, color='orange', label='Prediction')
                    plt.plot(input[0, :, -1], color='blue', label='Observations')
                    plt.legend()
                    
                    if self.number_of_actual_epochs == 0:
                        title = f"Model: {self.args.model}, Loss: {self.args.loss}, Epochs: {self.args.train_epochs}, W: {self.args.seq_len}, H: {self.args.pred_len}, Dataset: {self.args.data_path}"
                    elif self.number_of_actual_epochs > 0:
                        title = f"Model: {self.args.model}, Loss: {self.args.loss}, Epochs: {self.number_of_actual_epochs}, Max epochs: {self.args.train_epochs}, W: {self.args.seq_len}, H: {self.args.pred_len}, Dataset: {self.args.data_path}"
                    plt.title(title)
                    
                    plt.savefig(os.path.join(folder_path, 'Sample ' + str(i) + '.png'))
                    plt.close()
                    

        preds = np.array(preds)
        trues = np.array(trues)
        #print('preds and trues shapes:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        #print('preds and trues shapes passed through the compute_metrics function:', preds.shape, trues.shape)

        # result save
        #folder_path = './outputs/numerical_results/' + setting + '/'
        folder_path = './new_outputs/numerical_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        results = compute_metrics(preds, trues)

        # print('Gains (%) (from first to last epoch):')
        # for metric in self.list_of_metrics:
        #     print(f"{metric}:", self.gains_test_metrics[metric][-1])

        metrics = {metric: results[metric] for metric in self.list_of_metrics}

        formatted_metrics = ", ".join([f"{metric}:{metrics[metric]}" for metric in self.list_of_metrics])
        print(formatted_metrics)

        file_path = os.path.join(folder_path, f"txt_metrics_{setting}.txt")
        with open(file_path, 'a') as f:
            f.write(setting + "  \n")
            f.write(formatted_metrics + '\n')
            # f.write('Gains (%) (from first to last epoch):')
            # for metric in self.list_of_metrics:
            #     f.write(f"{metric}:", self.gains_test_metrics[metric][-1])
            f.write('\n')
            f.write('\n')

        np.save(folder_path + 'metrics.npy', np.array(list(results.values())))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return results

    def test_structural(self, setting, test=0):
        print('Beginning the test')

        test_data, test_loader = self._get_data(flag='test')
        structural_test_data, structural_test_loader = self._get_data_structural(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./new_outputs/checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        folder_path = './new_outputs/visual_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            for (i, (batch_x, batch_y, batch_x_mark, batch_y_mark)), (_, (structural_batch_x, structural_batch_y, structural_batch_x_mark, structural_batch_y_mark)) in zip(enumerate(test_loader), enumerate(structural_test_loader)):
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                structural_batch_y = structural_batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark) 
                
                f_dim = -1 if self.args.features == 'MS' else 0                       
                outputs = outputs[:, -self.args.pred_len:, f_dim:]                    
                #batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device) 
                structural_batch_y = structural_batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)   
                outputs = outputs.detach().cpu().numpy()                              
                #batch_y = batch_y.detach().cpu().numpy() 
                structural_batch_y = structural_batch_y.detach().cpu().numpy()                             
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    #batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                    structural_batch_y = test_data.inverse_transform(structural_batch_y.squeeze(0)).reshape(shape)
                
                pred = outputs
                #true = batch_y
                true = structural_batch_y

                preds.append(pred)
                trues.append(true)

                number_samples = 3
                number_batches = len(test_loader)
                plot_indices = [i * number_batches // number_samples for i in range(number_samples)]
                
                if i in plot_indices:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

                    plt.figure(figsize=(10, 5))
                    plt.plot(gt, color='darkblue', label='Ground Truth')
                    plt.plot(pd, color='orange', label='Prediction')
                    plt.plot(input[0, :, -1], color='blue', label='Observations')
                    plt.legend()
                    
                    if self.number_of_actual_epochs == 0:
                        title = f"Model: {self.args.model}, Loss: {self.args.loss}, Epochs: {self.args.train_epochs}, W: {self.args.seq_len}, H: {self.args.pred_len}, Dataset: {self.args.data_path}"
                    elif self.number_of_actual_epochs > 0:
                        title = f"Model: {self.args.model}, Loss: {self.args.loss}, Epochs: {self.number_of_actual_epochs}, Max epochs: {self.args.train_epochs}, W: {self.args.seq_len}, H: {self.args.pred_len}, Dataset: {self.args.data_path}"
                    plt.title(title)
                    
                    plt.savefig(os.path.join(folder_path, 'Sample ' + str(i) + '.png'))
                    plt.close()
           

        preds = np.array(preds)
        trues = np.array(trues)
        #print('preds and trues shapes:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        #print('preds and trues shapes passed through the compute_metrics function:', preds.shape, trues.shape)

        folder_path = './new_outputs/numerical_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        results = compute_metrics(preds, trues)

        metrics = {metric: results[metric] for metric in self.list_of_metrics}
        formatted_metrics = ", ".join([f"{metric}:{metrics[metric]}" for metric in self.list_of_metrics])
        print(formatted_metrics)

        file_path = os.path.join(folder_path, f"txt_metrics_{setting}.txt")
        with open(file_path, 'a') as f:
            f.write(setting + "  \n")
            f.write(formatted_metrics + '\n')
            f.write('\n')
            f.write('\n')

        np.save(folder_path + 'metrics.npy', np.array(list(results.values())))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return results

    def plot_batches_without_prediction(self, setting):

        test_data, test_loader = self._get_data(flag='test')

        dataset_name = self.args.data_path.split('.')[0]
        folder_path = './new_outputs/dataset_visualization/' + dataset_name + '/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                true = batch_y

                number_samples = 3
                number_batches = len(test_loader)
                plot_indices = [i * number_batches // number_samples for i in range(number_samples)]

                if i in plot_indices:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)

                    plt.figure(figsize=(10, 5))
                    plt.plot(gt, color='blue')
                    plt.plot(input[0, :, -1], color='blue')
                    
                    title = f"Sample (1D) of lenght: {self.args.seq_len + self.args.pred_len}, Dataset: {self.args.data_path.split('.')[0]}"
                    plt.title(title)
                    
                    plt.savefig(os.path.join(folder_path, 'Sample ' + str(i) + '.png'))
                    plt.close()


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            #path = os.path.join(self.args.checkpoints, setting)
            path = './new_outputs/checkpoints/' + setting
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        #folder_path = './outputs/numerical_results/' + setting + '/'
        folder_path = './new_outputs/numerical_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    
    def test_on_vali(self, setting, test=0):
        print('Beginning the test on validation dataset')
        vali_data, vali_loader = self._get_data(flag='val')
        if test:
            print('loading model')
            #self.model.load_state_dict(torch.load(os.path.join('./outputs/checkpoints/' + setting, 'checkpoint.pth')))
            self.model.load_state_dict(torch.load(os.path.join('./new_outputs/checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if vali_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                    

        preds = np.array(preds)
        trues = np.array(trues)
        
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # result save
        folder_path = './new_outputs/numerical_results/test_on_vali' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        results = compute_metrics(preds, trues)

        metrics = {metric: results[metric] for metric in self.list_of_metrics}

        formatted_metrics = ", ".join([f"{metric}:{metrics[metric]}" for metric in self.list_of_metrics])
        print(formatted_metrics)

        file_path = os.path.join(folder_path, f"txt_metrics_on_vali_{setting}.txt")
        with open(file_path, 'a') as f:
            f.write(setting + "  \n")
            f.write(formatted_metrics + '\n')
            f.write('\n')
            f.write('\n')

        np.save(folder_path + 'metrics.npy', np.array(list(results.values())))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return results

