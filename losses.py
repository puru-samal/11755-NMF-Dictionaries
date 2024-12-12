import torch
import torch.nn as nn

class BetaDivergenceLoss(nn.Module):
    '''
    PyTorch implementation of the beta-divergence loss function as a nn.Module
    '''
    def __init__(self, beta, reduction='sum'):
        super(BetaDivergenceLoss, self).__init__()
        if not (0 <= beta <= 1):
            raise ValueError("Beta must be between 0 and 1")
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError("reduction must be 'none', 'mean' or 'sum'")
        self.beta = beta
        self.reduction = reduction

    def forward(self, input, target):
        '''
        Input: input and target are 2D arrays of shape (n_samples, time_steps)
        Output: a scalar value if reduction is 'mean' or 'sum', 
                or tensor of shape (n_samples,) if reduction is 'none'
        '''
        if 0 < self.beta < 1:
            # Generalized Kullback-Leibler divergence
            loss = (1/(self.beta * (self.beta - 1))) * (
                target**self.beta + (self.beta - 1) * input**self.beta - 
                self.beta * target * input**(self.beta - 1)
            )
        elif self.beta == 0:
            # Itakura-Saito divergence
            loss = target / input - torch.log(target / input) - 1
        elif self.beta == 1: # beta == 1
            # KL divergence
            loss = target * torch.log(target / input) - target + input
        else:
            # Euclidean divergence
            loss = (target - input)**2

        if self.reduction == 'none':
            return loss.sum(dim=1)
        elif self.reduction == 'mean':
            return loss.mean()
        else: # reduction == 'sum'
            return loss.sum()


class SparsenessLoss(nn.Module):
    '''
    PyTorch implementation of the sparseness measure as a nn.Module
    '''
    def __init__(self, reduction='sum'):
        super(SparsenessLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError("reduction must be 'none', 'mean' or 'sum'")
        self.reduction = reduction

    def forward(self, input):
        '''
        Input: input is a 2D array of shape (n_samples, time_steps)
        Output: a scalar value if reduction is 'mean' or 'sum',
                or tensor of shape (n_samples,) if reduction is 'none'
        '''
        N = input.shape[1]  # number of time steps
        sparseness = (N ** 0.5 - (input.norm(p=1, dim=1)/ input.norm(p=2, dim=1)))/(N ** 0.5 - 1)
        
        if self.reduction == 'none':
            return sparseness
        elif self.reduction == 'mean':
            return sparseness.mean()
        else: # reduction == 'sum'
            return sparseness.sum()
