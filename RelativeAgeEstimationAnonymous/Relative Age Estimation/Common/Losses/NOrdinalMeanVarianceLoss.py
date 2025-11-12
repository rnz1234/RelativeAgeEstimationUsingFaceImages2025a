import torch
import torch.nn.functional as F
from torch import nn

# implementation
class OrdinalMeanVarianceLoss(nn.Module):
    def __init__(self, LamdaMean=0.2, LamdaVar=0.05, device=None, reduction = None):
        super(OrdinalMeanVarianceLoss, self).__init__()

        self.LamdaMean = torch.tensor(LamdaMean, requires_grad=False)
        self.LamdaVar  = torch.tensor(LamdaVar, requires_grad=False)
        self.reduction = reduction
        self.device    = device

        self.Sigmoid = nn.Sigmoid()
    def forward(self, input, target,IsProbability = False):

        Labels = torch.arange(input.shape[1]).float().cuda(input.device)

        if IsProbability == True:
            P = input
        else:
            P = self.Sigmoid(input)

        EstimatedMean = P.sum(1)

        #soft estimate per sample
        LossMean = ((torch.squeeze(EstimatedMean) - torch.squeeze(target).float()) ** 2)/2

        EX2 = 2*(P*Labels).sum(1)
        LossVar = F.relu(EX2-EstimatedMean**2)

        Result = self.LamdaMean*LossMean.mean() + self.LamdaVar*LossVar.mean()

        return Result