import torch


# Position kernels

def PositionGaussian(sigma_t, sigma_s, n_dim, dtype=torch.float, device="cpu"):
    sigmas = torch.ones((1,1,n_dim), dtype=dtype, device=device)
    sigmas[0,0,0] /= sigma_t
    sigmas[0,0,1:] /= sigma_s
    def K(bx,by):
        return torch.exp(-torch.sum(((sigmas*bx)[:,:,None,:] - (sigmas*by)[:,None,:,:])**2,dim=-1))
    return K

def PositionCauchy(sigma_t, sigma_s, n_dim, dtype=torch.float, device="cpu"):
    sigmas = torch.ones((1,1,n_dim), dtype=dtype, device=device)
    sigmas[0,0,0] /= sigma_t
    sigmas[0,0,1:] /= sigma_s
    def K(bx, by):
        diffs = sigmas * bx[:, :, None, :] - sigmas * by[:, None, :, :]
        squared_diffs = torch.sum(diffs ** 2, dim=-1)
        return 1 / (1 + squared_diffs)
    return K


# Orientation kernels

def OrientationDistribution(sigma_t, sigma_s, n_dim, dtype=torch.float, device="cpu"):
    sigmas = torch.ones((1,1,n_dim), dtype=dtype, device=device)
    sigmas[0,0,0] /= sigma_t
    sigmas[0,0,1:] /= sigma_s
    def K(bx, by):
        return torch.sum(torch.ones_like((sigmas*bx)[:,:,None,:] * (sigmas*by)[:,None,:,:]), axis=-1)
    return K

def OrientationCurrent(sigma_t, sigma_s, n_dim, dtype=torch.float, device="cpu"):
    sigmas = torch.ones((1,1,n_dim), dtype=dtype, device=device)
    sigmas[0,0,0] /= sigma_t
    sigmas[0,0,1:] /= sigma_s
    def K(bx,by):
        return torch.sum(((sigmas*bx)[:,:,None,:] * (sigmas*by)[:,None,:,:]),axis=-1)
    return K

def OrientationUnorientedVarifold(sigma_t, sigma_s, n_dim, dtype=torch.float, device="cpu"):
    sigmas = torch.ones((1,1,n_dim), dtype=dtype, device=device)
    sigmas[0,0,0] /= sigma_t
    sigmas[0,0,1:] /= sigma_s
    def K(bx,by):
        return torch.sum(((sigmas*bx)[:,:,None,:] * (sigmas*by)[:,None,:,:]),axis=-1)**2
    return K

def OrientationOrientedVarifold(sigma_t, sigma_s, n_dim, dtype=torch.float, device="cpu"):
    sigmas = torch.ones((1,1,n_dim), dtype=dtype, device=device)
    sigmas[0,0,0] /= sigma_t
    sigmas[0,0,1:] /= sigma_s
    def K(bx,by):
        return torch.exp(-torch.sum(((sigmas*bx)[:,:,None,:] - (sigmas*by)[:,None,:,:])**2,dim=-1))
    return K


# Propositions of adaptations to get close to the article => speak about this with Thibaut

# But even here, the definition of the scalar product is not exactly the same as in the article
# Notice that the scalar product problem discrepancy is also present on Current and Unoriented
# It is the same idea, but one have to choose one way
def OrientationOrientedVarifoldProposal(sigma_t, sigma_s, n_dim, dtype=torch.float, device="cpu"):
    sigmas = torch.ones((1,1,n_dim), dtype=dtype, device=device)
    sigmas[0,0,0] /= sigma_t
    sigmas[0,0,1:] /= sigma_s
    def K(bx,by):
        return torch.exp(2*torch.sum(((sigmas*bx)[:,:,None,:] * (sigmas*by)[:,None,:,:]), axis=-1))
    return K






# Definition of the kernel to define the loss

position_kernel_dictionary = {"Gaussian": PositionGaussian, "Cauchy": PositionCauchy}
orientation_kernel_dictionary = {"Distribution": OrientationDistribution, "Current": OrientationCurrent, "UnorientedVarifold": OrientationUnorientedVarifold, "OrientedVarifold": OrientationOrientedVarifold}

def OneKernel(position_kernel, orientation_kernel, sigma_t_pos, sigma_s_pos, 
              sigma_t_or, sigma_s_or, n_dim, dtype=torch.float, device="cpu"):
    K_position = position_kernel_dictionary[position_kernel](sigma_t, sigma_s, n_dim, dtype=dtype, device=device)
    K_orientation = orientation_kernel_dictionary[orientation_kernel](sigma_t, sigma_s, n_dim, dtype=dtype, device=device)
    def K(x,y,u,v):
        return K_position(x,y)*K_orientation(u,v)
    return K

def TwoKernels(position_kernel_little, orientation_kernel_little, position_kernel_big, orientation_kernel_big, 
               sigma_t_pos_little, sigma_s_pos_little, sigma_t_or_little, sigma_s_or_little, 
               sigma_t_pos_big, sigma_s_pos_big, sigma_t_or_big, sigma_s_or_big, 
               weight_little, weight_big, n_dim, dtype=torch.float, device="cpu"):
    K_position_little = position_kernel_dictionary[position_kernel_little](sigma_t_pos_little, sigma_s_pos_little, n_dim, dtype=dtype, device=device)
    K_orientation_little = orientation_kernel_dictionary[orientation_kernel_little](sigma_t_or_little, sigma_s_or_little, n_dim, dtype=dtype, device=device)
    K_position_big = position_kernel_dictionary[position_kernel_big](sigma_t_pos_big, sigma_s_pos_big, n_dim, dtype=dtype, device=device)
    K_orientation_big = orientation_kernel_dictionary[orientation_kernel_big](sigma_t_or_big, sigma_s_or_big, n_dim, dtype=dtype, device=device)*
    def K(x,y,u,v):
        K_position = weight_little*K_position_little(x,y) + weight_big*K_position_big(x,y)
        K_orientation = weight_little*K_orientation_little(u,v) + weight_big*K_orientation_big(u,v)
        return K_position*K_orientation
    return K


# Definition of the loss
    
def time_embed(bx, device="cpu"):
    B, T, _ = bx.shape
    time = torch.arange(T, device=device).view(1,-1,1).repeat_interleave(B,0).float()
    bx = bx.to(device)
    return torch.cat((time, bx),-1)

def compute_position_tangent_volume(tbx, device="cpu"):
    _, T, _ = tbx.shape
    indices0 = torch.arange(T-1, device=device)
    indices1 = torch.arange(1,T, device=device)
    tensor0 = torch.index_select(tbx, 1, indices0)
    tensor1 = torch.index_select(tbx, 1, indices1)
    position = 0.5*(tensor0 + tensor1)
    tangent = tensor1 - tensor0
    volume = torch.norm(tangent, p=2, dim=-1)
    tangent = tangent / volume.unsqueeze(-1)
    return position, tangent, volume

def VarifoldLoss(K,reduction = "mean", device="cpu"):
    def loss(bx,by):
        B, T, C = bx.shape
        tbx,tby = time_embed(bx, device=device),time_embed(by, device=device)
        px,tx,vx = compute_position_tangent_volume(tbx, device=device)
        py,ty,vy = compute_position_tangent_volume(tby, device=device)
        c0 = torch.sum(vx[:,:,None] * K(px,px,tx,tx) * vx[:,None,:],dim =(1,2))
        c1 = torch.sum(vx[:,:,None] * K(px,py,tx,ty) * vy[:,None,:],dim =(1,2))
        c2 = torch.sum(vy[:,:,None] * K(py,py,ty,ty) * vy[:,None,:],dim =(1,2))
        blosses =  c0 -2*c1 + c2
        if reduction == "mean":
            tensor_loss = torch.sum(blosses)/(B*(T-1)*C)
            return tensor_loss.cpu()
        if reduction == "sum":
            tensor_loss = torch.sum(blosses)
            return tensor_loss.cpu()
        if reduction == "none":
            tensor_loss = blosses
            return tensor_loss.cpu()
    return loss