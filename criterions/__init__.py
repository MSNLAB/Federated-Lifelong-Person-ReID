from criterions.cross_entropy import CrossEntropyLabelSmooth
from criterions.triplet_loss import TripletLoss

criterions = {
    'cross_entropy': CrossEntropyLabelSmooth,
    'triplet_loss': TripletLoss,
}
