import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self, multiplier=1.0):
        super(SoftTargetCrossEntropy, self).__init__()
        self.multiplier = multiplier

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(self.multiplier*x, dim=-1), dim=-1)
        return loss.mean()

# class SoftTargetCrossEntropy_Smooth(nn.Module):
#
#     def __init__(self):
#         super(SoftTargetCrossEntropy_Smooth, self).__init__()
#
#     def forward(self, x, target):
#         model_outputs = F.log_softmax(x, dim=-1)
#         model_output_prob_T = model_outputs ** (1/0.5)
#         model_output_prob_sharping = model_output_prob_T/model_output_prob_T.sum(dim=-1, keepdim=True)
#         if target.shape != x.shape:
#             assert len(target.shape)==1 and target.shape[0] == x.shape[0]
#             target_onehot = torch.nn.functional.one_hot(target, 100).type(torch.cuda.FloatTensor)
#             loss = torch.sum(-target_onehot * model_output_prob_sharping, dim=-1)
#         else:
#             loss = torch.sum(-target * model_output_prob_sharping, dim=-1)
#         return loss.mean()

# from torch.nn.modules.loss import _WeightedLoss
# class CrossEntropyLoss_Smooth(_WeightedLoss):
#     __constants__ = ['weight', 'ignore_index', 'reduction']
#
#     def __init__(self, weight=None, size_average=None, ignore_index=-100,
#                  reduce=None, reduction='mean'):
#         super(CrossEntropyLoss_Smooth, self).__init__(weight, size_average, reduce, reduction)
#         self.ignore_index = ignore_index
#
#     def forward(self, input, target):
#         return F.cross_entropy(input, target, weight=self.weight,
#                                ignore_index=self.ignore_index, reduction=self.reduction)