import torch
import torch.nn as nn
import torch.nn.functional as F


def LossFunction(loss_type):
    if loss_type == 'CrossEntropy':
        loss_fn = SampledCrossEntropyLoss()
    elif loss_type == 'TOP1':
        loss_fn = TOP1Loss()
    elif loss_type == 'BPR':
        loss_fn = BPRLoss()
    elif loss_type == 'TOP1-max':
        loss_fn = TOP1_max()
    elif loss_type == 'BPR-max':
        loss_fn = BPR_max()
    else:
        raise NotImplementedError
    return loss_fn


class SampledCrossEntropyLoss(nn.Module):
    """ CrossEntropyLoss with n_classes = batch_size = the number of samples in the session-parallel mini-batch """
    def __init__(self):
        super(SampledCrossEntropyLoss, self).__init__()
        self.xe_loss = nn.CrossEntropyLoss()

    def forward(self, logit):
        batch_size = logit.size(1)

        target = torch.arange(batch_size).long().to(logit.device)
        return self.xe_loss(logit, target)


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to sampled number of items to evaluate
        """
        # differences between the item scores
        diff = logit.diag().view(-1, 1).expand_as(logit) - logit
        # final loss
        loss = -torch.mean(F.logsigmoid(diff))
        return loss


class BPR_max(nn.Module):
    def __init__(self):
        super(BPR_max, self).__init__()
    def forward(self, logit):
        logit_softmax = F.softmax(logit, dim=1)
        diff = logit.diag().view(-1, 1).expand_as(logit) - logit
        loss = -torch.log(torch.mean(logit_softmax * torch.sigmoid(diff)))
        return loss


class TOP1Loss(nn.Module):
    def __init__(self):
        super(TOP1Loss, self).__init__()
    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to sampled number of items to evaluate
        """
        diff = -(logit.diag().view(-1, 1).expand_as(logit) - logit)
        loss = torch.sigmoid(diff).mean() + torch.sigmoid(logit ** 2).mean()
        return loss


class TOP1_max(nn.Module):
    def __init__(self):
        super(TOP1_max, self).__init__()

    def forward(self, logit):
        logit_softmax = F.softmax(logit, dim=1)
        diff = -(logit.diag().view(-1, 1).expand_as(logit) - logit)
        loss = torch.mean(logit_softmax * (torch.sigmoid(diff) + torch.sigmoid(logit ** 2)))
        return loss