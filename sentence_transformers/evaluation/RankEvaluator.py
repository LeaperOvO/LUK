from . import SentenceEvaluator
import torch
from torch.utils.data import DataLoader
import logging
from ..util import batch_to_device
import os
import csv
import sklearn
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,confusion_matrix


logger = logging.getLogger(__name__)

class RankEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on its accuracy on a labeled dataset

    This requires a model with LossFunction.SOFTMAX

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, dataloader: DataLoader, name: str = "", softmax_model = None, write_csv: bool = True):
        """
        Constructs an evaluator for the given dataset

        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        self.name = name
        self.softmax_model = softmax_model

        if name:
            name = "_"+name

        self.write_csv = write_csv
        self.csv_file = "accuracy_evaluation"+name+"_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        total = 0
        correct = 0
        y_pred = []
        y_true = []
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"
        result = torch.zeros((len(self.dataloader),16)).to(model.device)

        logger.info("Evaluation on the "+self.name+" dataset"+out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate
        for step, batch in enumerate(self.dataloader):
            features, label_ids = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            label_ids = label_ids.to(model.device)
            with torch.no_grad():
                _, prediction = self.softmax_model(features, labels=None)
            result[step] = torch.softmax(prediction, dim=1)[:,1]

        result = 1 - result
        _, indices = torch.sort(result, dim=1)
        _, indices = torch.sort(indices, dim=1)

        MRR = 0
        p1 = 0
        p3 = 0
        result = indices.cpu().numpy().tolist()
        for i in range(len(result)):
            if result[i][0] == 0:
                p1 += 1
            if result[i][0] <= 2:
                p3 += 1
            MRR += (1.0 / (result[i][0] + 1))
        print('p1:' , p1 / len(result))
        print('p3:' , p3 / len(result))
        print('mrr:' ,MRR / len(result))


        return 1
