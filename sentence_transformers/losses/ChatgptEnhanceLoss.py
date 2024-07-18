import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Union, Tuple, List, Iterable, Dict, Callable
from ..SentenceTransformerEnhance import SentenceTransformerEnhance
import logging
from sentence_transformers import util
import math
import os
from .. import util

logger = logging.getLogger(__name__)
class ChatgptEnhanceLoss(nn.Module):
    def __init__(self,
                 model: SentenceTransformerEnhance,
                 sentence_embedding_dimension: int,
                 loss_fct: Callable = nn.CrossEntropyLoss(ignore_index=-100),scale: float = 20.0, similarity_fct = util.cos_sim):
        super(ChatgptEnhanceLoss, self).__init__()
        self.model = model
        self.sentence_embedding_dimension = sentence_embedding_dimension
        self.mlm = nn.Linear(sentence_embedding_dimension * 2, len(self.model._first_module().tokenizer))
        self.q = nn.Linear(sentence_embedding_dimension, sentence_embedding_dimension)
        self.k = nn.Linear(sentence_embedding_dimension, sentence_embedding_dimension)
        self.v = nn.Linear(sentence_embedding_dimension, sentence_embedding_dimension)

        self.sentence_embedding_dimension = sentence_embedding_dimension
        self.loss_fct = loss_fct
        self.dk = torch.sqrt(torch.FloatTensor([self.sentence_embedding_dimension])).to(self.model._target_device)
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(0.1)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: [Dict, Tensor]):
        # reps = [self.model(sentence_feature)['token_embeddings'] for sentence_feature in sentence_features]
        embeddings_a = self.model(sentence_features[0])['token_embeddings']
        embeddings_b = self.model(sentence_features[1])['token_embeddings']
        Q = self.q(embeddings_a)
        K = self.k(embeddings_b)
        V = self.v(embeddings_b)

        energy = torch.matmul(Q,K.permute(0,2,1)) / self.dk
        attention_mask = sentence_features[1]["attention_mask"]
        mask = attention_mask.unsqueeze(1).repeat(1,Q.size()[1],1)
        energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)

        confusion_input = torch.cat((embeddings_a, x), dim=2)
        mlm = self.mlm(confusion_input)
        mlm_label = labels["mlm_label"].to(self.model._target_device)
        loss_mlm = self.loss_fct(mlm.view(-1, len(self.model._first_module().tokenizer)), mlm_label.view(-1))


        se_a = embeddings_a[:,0,:]
        se_b = embeddings_b[:,0,:]

        scores = self.similarity_fct(se_a, se_b) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long,
                              device=scores.device)  # Example a[i] should match with b[i]
        return loss_mlm + self.cross_entropy_loss(scores, labels)
