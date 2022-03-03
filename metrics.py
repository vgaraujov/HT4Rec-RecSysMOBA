import torch
import numpy as np


def get_recall(indices, targets):
    """ Calculates the recall score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        recall (float): the recall score
    """
    targets = targets.view(-1, 1).expand_as(indices)  # (Bxk)
    hits = (targets == indices).nonzero()
    if len(hits) == 0: return 0, 0
    n_hits = (targets == indices).nonzero()[:, :-1].size(0)
    recall = n_hits / targets.size(0)
    precision = (n_hits / targets.size(1)) / targets.size(0)

    return recall, precision

def get_mrr(indices, targets):
    """ Calculates the MRR score for the given predictions and targets
    
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        mrr (float): the mrr score
    """
    targets = targets.view(-1,1).expand_as(indices)
    # ranks of the targets, if it appears in your indices
    hits = (targets == indices).nonzero()
    if len(hits) == 0: return 0
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)  # reciprocal ranks
    mrr = torch.sum(rranks).data / targets.size(0)
    mrr = mrr.item()
    
    return mrr

def get_ap(indices, targets):
    """ Calculates the MRR score for the given predictions and targets
    
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        mrr (float): the mrr score
    """
    scores = []
    for preds, tgts in zip(indices, targets):
        score = 0.0
        num_hits = 0.0
        for i, p in enumerate(preds):
            if p in tgts and p not in preds[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        scores.append(score)
    ap = sum(scores)/targets.size(0)
    
    return ap

def evaluate(logits, targets, k=20):
    """ Evaluates the model using Recall@K, MRR@K scores.
    Args:
        logits (B,C): torch.LongTensor. The predicted logit for the next items.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        recall (float): the recall score
        mrr (float): the mrr score
    """
    
    _, indices = torch.topk(logits, k, -1)
    recall, precision = get_recall(indices, targets)
    mrr = get_mrr(indices, targets)
    ap = get_ap(indices, targets)
    
    return recall, precision, ap, mrr
