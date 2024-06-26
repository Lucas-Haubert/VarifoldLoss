import torch

def TSGaussKernel(sigma_t,sigma_s,n_dim,dtype = torch.float,device = "cpu"):
    # ATTENTION: n_dim is the dimension of the time + space embedding. If the signal has N dimension n_dim shoulb be N+1.
    sigmas = torch.ones((1,1,n_dim),dtype = dtype, device = device)
    sigmas[0,0,0] /= sigma_t
    sigmas[0,0,1:] /= sigma_s
    def K(bx,by):
        return torch.exp(-torch.sum(((sigmas*bx)[:,:,None,:] - (sigmas*by)[:,None,:,:])**2,dim=-1))
    return K
def TSGaussGaussKernel(sigma_t_1,sigma_s_1,sigma_t_2,sigma_s_2,n_dim,dtype = torch.float,device = "cpu"):
    # ATTENTION: n_dim is the dimension of the time + space embedding. If the signal has N dimension n_dim shoulb be N+1.
    K1 = TSGaussKernel(sigma_t_1,sigma_s_1,n_dim=n_dim,dtype=dtype,device=device)
    K2 = TSGaussKernel(sigma_t_2,sigma_s_2,n_dim=n_dim,dtype=dtype,device=device)
    def K(x,y,u,v):
        return K1(x,y)*K2(u,v)
    return K
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
    # It defines a batch loss
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
            return torch.sum(blosses)/(B*(T-1)*C)
        if reduction == "sum":
            return torch.mean(blosses)
        if reduction == "none":
            return blosses
    return loss