import torch
from torch import nn
from torch.autograd import grad
import numpy as np
from tianshou.data import to_torch

effective_dim_start = 3
effective_dim_end = 8

def get_model(model_file):#, use_cuda = False):
    n = 8
    m = 3
    dim = effective_dim_end - effective_dim_start # 8 - 3. Do not use the positions
    c = 3 * n
    model_u_w1 = torch.nn.Sequential(
        torch.nn.Linear(2*dim, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, c*n, bias=True))

    model_u_w2 = torch.nn.Sequential(
        torch.nn.Linear(2*dim, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, m*c, bias=True))

    # if use_cuda:
    #     model_u_w1 = model_u_w1.cuda()
    #     model_u_w2 = model_u_w2.cuda()

    ck = torch.load(model_file)
    model_u_w1.load_state_dict(ck['model_u_w1'])
    model_u_w2.load_state_dict(ck['model_u_w2'])

    def u_func(x, xe, uref):
        x = torch.from_numpy(x).float().view(1,-1,1)
        xe = torch.from_numpy(xe).float().view(1,-1,1)
        uref = torch.from_numpy(uref).float().view(1,-1,1)

        # x: B x n x 1
        # u: B x m x 1
        bs = x.shape[0]

        w1 = model_u_w1(torch.cat([x[:,effective_dim_start:effective_dim_end,:],(x-xe)[:,effective_dim_start:effective_dim_end,:]],dim=1).squeeze(-1)).reshape(bs, -1, n)
        w2 = model_u_w2(torch.cat([x[:,effective_dim_start:effective_dim_end,:],(x-xe)[:,effective_dim_start:effective_dim_end,:]],dim=1).squeeze(-1)).reshape(bs, m, -1)
        u = w2.matmul(torch.tanh(w1.matmul(xe))) + uref

        u = u.squeeze(0).detach().numpy()
        return u

    return u_func
