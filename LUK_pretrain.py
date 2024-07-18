import argparse
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformerEnhance, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator,LabelAccuracyEvaluator,BinaryClassificationEvaluator
from torch.utils.data import DataLoader
import logging
import json
import random
import os
import sys
import math
import numpy as np

random.seed(1)
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-pretrain_data", "--pretrain_data", type=str, help="pre-train data directory")

    args.add_argument("-base_model", "--base_model", type=str,
                      default="bert-base-uncased", help="base_model")

    args.add_argument("-epoch", "--epoch", type=int,
                      default=50, help="Number of epochs")

    args.add_argument("-batch_size", "--batch_size", type=int,
                      default=8, help="Batch Size")

    args.add_argument("-outfolder", "--outfolder", type=str,
                      default="./output/LUK_chatgpt_enchance", help="Folder name to save the models.")

    args = args.parse_args()
    return args

def read_json(file):
    with open(file, 'r+') as file:
        content = file.read()
    content = json.loads(content)
    return content

def train(args):
    datapath = args.pretrain_data
    model_save_path = args.outfolder
    train_batch_size = args.batch_size
    num_epochs = args.epoch

    # load data
    data = read_json(datapath)

    # load model
    model_name = args.base_model

    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformerEnhance(modules=[word_embedding_model, pooling_model])

    # clean data
    data_clean = []
    for item in data:
        if item[1] != "":
            data_clean.append(item)
    print(data_clean[0])
    train = []
    for item in data_clean:
        train.append(InputExample(texts=[item[0], item[1]]))

    # load data
    train_dataloader = DataLoader(train, shuffle=True, batch_size=train_batch_size)

    # pre-train loss: TP + SA
    train_loss = losses.ChatgptEnhanceLoss(model=model,
                                           sentence_embedding_dimension=model.get_sentence_embedding_dimension())
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # train
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=num_epochs,
              warmup_steps=warmup_steps,
              output_path=model_save_path,
              optimizer_params={'lr': 5e-5},
              )

if __name__ == '__main__':
    args = parse_args()
    train(args)
