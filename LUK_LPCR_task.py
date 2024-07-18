import argparse
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import RankEvaluator
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
    args.add_argument("-data", "--train_data", type=str,
                      default="./datasets/tasks/LPCR/hw_switch_causes.json", help="dataset")



    args.add_argument("-pretrain_model", "--pretrain_model", type=str,
                      default="bert-base-uncased", help="the path of the pretrained model to finetune")


    args.add_argument("-epoch", "--epoch", type=int,
                      default=10, help="Number of epochs")

    args.add_argument("-batch_size", "--batch_size", type=int,
                      default=8, help="Batch Size")

    args.add_argument("-outfolder", "--outfolder", type=str,
                      default="./output/luk_lpcr_finetune", help="Folder name to save the models.")

    args = args.parse_args()
    return args

def read_json(file):
    with open(file, 'r+') as file:
        content = file.read()
    content = json.loads(content)
    return content

def evaluate(args):
    model_save_path = args.outfolder
    train_batch_size = args.batch_size
    num_epochs = args.epoch

    data = read_json(args.train_data)

    # load model
    model = SentenceTransformer(args.pretrain_model)

    # load dataset
    x = []
    y = []
    for i in range(len(data)):
        data[i][0] = data[i][0].replace('\\"', '')
        x.append([data[i], 1])
        y.append(1)
        if i != len(data) - 1:
            neg_causes = data[i + 1][1]
        else:
            neg_causes = data[0][1]
        x.append([[data[i][0], neg_causes], 0])
        y.append(0)

    random.shuffle(x)

    choices = []
    for item in data:
        choices.append([item[1]])

    train_size = int(len(x) * 0.6)
    print(train_size)
    dev_size = int(len(x) * 0.8)

    test_result = []
    train_samples = []
    dev_samples = []
    test_samples = []
    for i in range(len(x)):
        if i <= train_size:
            train_samples.append(InputExample(texts=[x[i][0][0], x[i][0][1]], label=x[i][1]))
        elif i <= dev_size:
            dev_samples.append(InputExample(texts=[x[i][0][0], x[i][0][1]], label=x[i][1]))
        else:
            each_line = []
            if x[i][1] == 1:
                test_samples.append(InputExample(texts=[x[i][0][0], x[i][0][1]], label=x[i][1]))
                each_line.append(x[i][0][0])
                each_line.append(x[i][0][1])
                for j in range(15):
                    c = random.choice(choices)
                    test_samples.append(InputExample(texts=[x[i][0][0], c], label=0))
                    each_line.append(c)
            if len(each_line) != 0:
                test_result.append(each_line)

    # loss
    train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                    num_labels=2)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

    test_dataloader = DataLoader(test_samples, shuffle=False, batch_size=16)
    print(len(test_dataloader))

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    test_evaluator = RankEvaluator(test_dataloader, softmax_model=train_loss, name='test2')

    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator2=test_evaluator,
              epochs=num_epochs,
              evaluation_steps=10000,
              warmup_steps=warmup_steps,
              output_path=model_save_path,
              )

if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
