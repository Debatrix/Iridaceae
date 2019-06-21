from torch import nn
from torch.nn import functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin, metrics='euclidean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.metrics = metrics
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        if self.metrics == 'cosine':
            distances = F.cosine_similarity(output1, output2)
        else:
            distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, metrics='euclidean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.metrics = metrics

    def forward(self, anchor, positive, negative, size_average=True):
        if self.metrics == 'cosine':
            distance_positive = F.cosine_similarity(anchor, positive)
            distance_negative = F.cosine_similarity(anchor, negative)
        else:
            distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
            distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
