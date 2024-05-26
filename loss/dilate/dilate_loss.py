import torch

import loss.dilate.soft_dtw as soft_dtw
import loss.dilate.path_soft_dtw as path_soft_dtw


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def DILATE(outputs, targets, alpha=0.5, gamma=0.01, device=device):
	# outputs, targets: shape (B, H, C_in) ; H is noted N_outputs here
	batch_size, N_output = outputs.shape[0:2]

	softdtw_batch = soft_dtw.SoftDTWBatch.apply
	D = torch.zeros((batch_size, N_output,N_output )).to(device)
	for k in range(batch_size):
		Dk = soft_dtw.pairwise_distances(targets[k,:,:],outputs[k,:,:])
		D[k:k+1,:,:] = Dk     
	loss_shape = softdtw_batch(D,gamma)

	path_dtw = path_soft_dtw.PathDTWBatch.apply
	path = path_dtw(D,gamma)       
	Omega =  soft_dtw.pairwise_distances(torch.arange(1,N_output+1).view(N_output,1)).to(device) 
	loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output) 

	loss = alpha * loss_shape + (1 - alpha) * loss_temporal

	return loss.cpu() # Similar to the output of nn.MSELoss()


def DILATE_independent(outputs, targets, alpha=0.5, gamma=0.01, device=device):
    # outputs, targets: shape (B, H, C_in)
    batch_size, forecast_horizon, num_channels = outputs.shape

    total_loss = 0

    for c in range(num_channels):
        outputs_channel = outputs[:, :, c].unsqueeze(-1)
        targets_channel = targets[:, :, c].unsqueeze(-1)

        loss = DILATE(outputs_channel, targets_channel, alpha, gamma, device)
        total_loss += loss

    result = total_loss / num_channels

    return result.cpu() # Similar to the output of nn.MSELoss()

