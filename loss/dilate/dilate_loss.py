import torch
import soft_dtw
import path_soft_dtw 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dilate_loss(outputs, targets, alpha, gamma, device):

	batch_size, N_output = outputs.shape[0:2]

	loss_shape = 0
	softdtw_batch = soft_dtw.SoftDTWBatch.apply
	D = torch.zeros((batch_size, N_output,N_output )).to(device)
	for k in range(batch_size):
		Dk = soft_dtw.pairwise_distances(targets[k,:,:],outputs[k,:,:])
		D[k:k+1,:,:] = Dk     
	loss_shape = softdtw_batch(D,gamma)

	path_dtw = path_soft_dtw.PathDTWBatch.apply
	path = path_dtw(D,gamma)       
	Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device) 
	loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output) 

	loss = alpha * loss_shape + (1 - alpha) * loss_temporal

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

outputs = torch.randn(2, 10, 3)
targets = torch.randn(2, 10, 3)

print("outputs:")
print(outputs)

print("\ntargets")
print(targets)


alpha = 0.5
gamma = 0.5


print("\nDILATE_multi_dim_naive(outputs, targets) with alpha =", alpha, " and gamma =", gamma)
start_time_naive = time.time()
loss_naive = DILATE_multi_dim_naive(outputs, targets, alpha, gamma, device)
end_time_naive = time.time()
duration_naive = end_time_naive - start_time_naive
print("Loss naive:", loss_naive)
print("Duration naive:", duration_naive, "seconds")

print("\ndilate_loss(outputs, targets) with alpha =", alpha, " and gamma =", gamma)
start_time_dilate_loss = time.time()
loss_dilate_loss = dilate_loss(outputs, targets, alpha, gamma, device)
end_time_dilate_loss = time.time()
duration_dilate_loss = end_time_dilate_loss - start_time_dilate_loss
print("Loss dilate original:", loss_dilate_loss)
print("Duration dilate original:", duration_dilate_loss, "seconds")



"""
print("Pairwise matrix on multivariate time series")
batch_size, N_output = outputs.shape[0:2]
D = torch.zeros((batch_size, N_output, N_output)).to(device)
for k in range(batch_size):
	Dk = soft_dtw.pairwise_distances(outputs[k,:,:],targets[k,:,:])
	D[k:k+1,:,:] = Dk  
print(D)   
softdtw_batch = soft_dtw.SoftDTWBatch.apply
print("Loss shape")
loss_shape = softdtw_batch(D,gamma)
print(loss_shape)"""


"""
print("\ndilate_loss(outputs, targets, alpha, gamma, device) with alpha =", alpha, " and gamma =", gamma)
loss, loss_shape, loss_temporal = dilate_loss(pred_channel, true_channel, alpha, gamma, device)
print(loss.item())
"""
