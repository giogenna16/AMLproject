import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as linalg

#Loss to mininize the negative entropy
class NegHLoss(nn.Module):
    def __init__(self):
        super(NegHLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum() #we want to mininize the negative entropy
        return b

#Loss for the reconstructor
class ReconstructionLoss(nn.Module):
    def __init__(self):
         super(ReconstructionLoss, self).__init__()
    
    def forward(self, x, y):
        #this code is to compute the first term
        output1= linalg.norm((x-y), ord="fro")**2

        #this code is to compute the second term
        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        log_x = F.log_softmax(x, dim=1) # maybe dim=0
        log_y = F.log_softmax(y, dim=1) # maybe dim=0
        output2 = kl_loss(log_x, log_y) #this is the second term

        return output1 + output2

