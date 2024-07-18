import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict, Callable
from ..SentenceTransformer import SentenceTransformer
import logging


logger = logging.getLogger(__name__)

class SingleSoftmaxLoss(nn.Module):

    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 loss_fct: Callable = nn.CrossEntropyLoss()):
        super(SingleSoftmaxLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels

        self.classifier = nn.Linear(sentence_embedding_dimension, num_labels)
        self.loss_fct = loss_fct

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_a = reps[0]
        output = self.classifier(rep_a)
        if labels is not None:
            loss = self.loss_fct(output, labels.view(-1))
            return loss
        else:
            return reps, output