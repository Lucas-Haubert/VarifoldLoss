import torch
import soft_dtw
import path_soft_dtw 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dilate_loss(outputs, targets, alpha, gamma, device):
	# outputs, targets: shape (batch_size, forecast horizon, 1)
	batch_size, N_output = outputs.shape[0:2]
	loss_shape = 0
	softdtw_batch = soft_dtw.SoftDTWBatch.apply
	D = torch.zeros((batch_size, N_output,N_output )).to(device)
	for k in range(batch_size):
		Dk = soft_dtw.pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1))
		D[k:k+1,:,:] = Dk     
	loss_shape = softdtw_batch(D,gamma)
	
	path_dtw = path_soft_dtw.PathDTWBatch.apply
	path = path_dtw(D,gamma)           
	Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device)
	loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output) 
	loss = alpha*loss_shape+ (1-alpha)*loss_temporal
	return loss, loss_shape, loss_temporal


def DILATE_multi_dim_naive(outputs, targets, alpha, gamma, device):
    # outputs, targets: shape (B, H, C_in)
    batch_size, forecast_horizon, num_channels = outputs.shape
    total_loss = 0.0

    for c in range(num_channels):
        outputs_channel = outputs[:, :, c].unsqueeze(-1)
        targets_channel = targets[:, :, c].unsqueeze(-1)

        loss, loss_shape, loss_temporal = dilate_loss(outputs_channel, targets_channel, alpha, gamma, device)
        total_loss += loss.item() 

    return total_loss / num_channels


# Tests

import time

pred_channel = torch.randn(2, 10, 3)
true_channel = torch.randn(2, 10, 3)

print("pred_channel:")
print(pred_channel)

print("\ntrue_channel")
print(true_channel)


alpha = 0.5
gamma = 0.5


print("\nDILATE_multi_dim_naive(pred, true) with alpha =", alpha, " and gamma =", gamma)
start_time_naive = time.time()
loss_naive = DILATE_multi_dim_naive(pred_channel, true_channel, alpha, gamma, device)
end_time_naive = time.time()
duration_naive = end_time_naive - start_time_naive
print("Loss naive:", loss_naive)
print("Duration naive:", duration_naive, "seconds")

"""print("\nDILATE_multi_dim_efficient(pred, true) with alpha =", alpha, " and gamma =", gamma)
start_time_efficient = time.time()
loss_efficient = DILATE_multi_dim_efficient(pred_channel, true_channel, alpha, gamma, device)
end_time_efficient = time.time()
duration_efficient = end_time_efficient - start_time_efficient
print("Loss efficient:", loss_efficient)
print("Duration efficient:", duration_efficient, "seconds")"""