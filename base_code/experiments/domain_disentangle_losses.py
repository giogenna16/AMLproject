import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as linalg



# Loss to mininize the negative entropy
class NegHLoss(nn.Module):
    def __init__(self):
        super(NegHLoss, self).__init__()

    # Mathematical definition given in the paper
    def forward(self, x):
        # get probabilities of logits to avoid numerical issues
        b = F.log_softmax(x + 1e-6, dim=1)
        # sum over the number of samples of given class and compute mean
        b = b.sum(dim=0) / x.size(0)  # we want to minimize the negative entropy
        # sum over the number of classes
        b = b.sum()
        return - b


# Loss for the reconstructor
class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
    
    def forward(self, x, y):
        # compute the first term (Frobenius Norm squared)
        output1 = linalg.norm((x-y), ord="fro")**2

        # compute the reconstruction loss regularization term
        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        log_x = F.log_softmax(x, dim=1)  # maybe dim=0
        log_y = F.log_softmax(y, dim=1)  # maybe dim=0
        output2 = kl_loss(log_x, log_y)  # this is the second term

        return output1 + output2

