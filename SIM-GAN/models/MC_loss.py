from packaging import version
import torch
from torch import nn
import math
from .sinkhorn import OT
import numpy as np
import torch.nn.functional as F

# class Normalize(nn.Module):

#     def __init__(self, power=2):
#         super(Normalize, self).__init__()
#         self.power = power

#     def forward(self, x):
#         norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
#         out = x.div(norm + 1e-7)
#         return out

class MC_Loss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        #self.l2_norm = Normalize(2)

    def forward(self, feat_src, feat_tgt, feat_gen):
        batchSize = feat_src.shape[0]
        dim = feat_src.shape[1]   
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size // len(self.opt.gpu_ids)

        # if self.loss_type == 'MoNCE':
        ot_src = feat_src.view(batch_dim_for_bmm, -1, dim).detach()
        ot_tgt = feat_tgt.view(batch_dim_for_bmm, -1, dim).detach()
        ot_gen = feat_gen.view(batch_dim_for_bmm, -1, dim)
        #print("ot_src:",ot_src.shape)
        #print("ot_tgt:",ot_tgt.shape)
        #print("ot_gen:",ot_gen.shape)
        f1 = OT(ot_src, ot_tgt, eps=self.opt.eps, max_iter=50, cost_type=self.opt.cost_type)
        #print("F1:",f1)
        f2 = OT(ot_tgt, ot_gen, eps=self.opt.eps, max_iter=50, cost_type=self.opt.cost_type)
        #print("F2:",f2)
        
        MC_Loss = F.l1_loss(f1, f2)
        return MC_Loss