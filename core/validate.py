import csv
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import models
import params
from module.torch import evaluate
from module.torch.Logger import Logger


class Validate:
    def __init__(self, logger: Logger, model_path: str):
        self.logger = logger
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = models.Encoder(n_channels=params.input_channel)
        self.decoder1 = models.Decoder(n_classes=params.num_class)
        self.decoder2 = models.Decoder(n_classes=params.num_class)
        self.build_model()

    def build_model(self):
        self.encoder.load_state_dict(torch.load(os.path.join(self.model_path, params.encoder_filename)))
        self.decoder1.load_state_dict(torch.load(os.path.join(self.model_path, params.decoder1_filename)))
        self.decoder2.load_state_dict(torch.load(os.path.join(self.model_path, params.decoder2_filename)))
        self.encoder.to(self.device)
        self.decoder1.to(self.device)
        self.decoder2.to(self.device)
        self.encoder.eval()
        self.decoder1.eval()
        self.decoder2.eval()

    def validate(self, data_loaders: list):
        criterion = nn.CrossEntropyLoss()

        for dataloader in data_loaders:
            losses = []
            accuracies = []
            with torch.no_grad():
                for dataset in tqdm(dataloader):
                    images_list = [
                        dataset['image1'].to(self.device),
                        dataset['image2'].to(self.device),
                        dataset['image3'].to(self.device)
                    ]
                    label = dataset['label2'].to(self.device)

                    # tgt t-1, t
                    feat_tgt_1 = self.encoder(images_list[0], images_list[1])
                    predict2 = self.decoder2(feat_tgt_1[1], feat_tgt_1[2])

                    # tgt t, t+1
                    feat_tgt_2 = self.encoder(images_list[1], images_list[2])
                    predict1 = self.decoder1(feat_tgt_2[0], feat_tgt_2[2])

                    predict = predict1 + predict2
                    loss = criterion(predict, label)
                    # predict = (predict1 + predict2).cpu().numpy().reshape(-1)
                    # label = label.cpu().numpy().reshape(-1)
                    # acc = evaluate.dice_accuracy(predict1 + predict2, label, mode='none', ignore_id=[0])
                    # acc = evaluate.iou_metrics(predict, label, mode='none', ignore_id=0)
                    acc = evaluate.recall_metrics(predict, label, mode='none', ignore_id=[0])
                    # acc = evaluate.categorical_accuracy(predict, label)

                    losses.append(loss.item())
                    accuracies.append(acc)

            with open(os.path.join(self.logger.log_path, "evaluate.log"), 'a') as log:
                scores = np.hstack([np.asarray(losses)[:, np.newaxis], np.reshape(accuracies, (len(losses), -1))])
                writer = csv.writer(log)
                writer.writerow(["loss", "acc"])
                writer.writerows(scores)
                writer.writerow([])

