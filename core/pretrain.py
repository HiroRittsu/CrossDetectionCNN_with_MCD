import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import models
from module.torch.evaluate import iou_metrics
import params


class PreTrain:
    def __init__(self, logger):
        if not os.path.exists(params.snapshot_path):
            os.makedirs(params.snapshot_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.encoder = models.Encoder(img_ch=params.input_channel).to(self.device).apply(models.init_weights)
        self.decoder = models.Decoder(output_ch=params.num_class).to(self.device).apply(models.init_weights)

    def train(self, src_train_loader, test_loader, tgt_test_loader):
        data_loaders = {'train': src_train_loader, 'src_val': test_loader, 'tgt_val': tgt_test_loader}
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=params.pre_learning_rate
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(params.pre_num_epochs):
            print('\nEpoch {}/{}'.format(epoch + 1, params.pre_num_epochs))
            print('-------------')
            losses = {key: [] for key in data_loaders.keys()}
            accuracies = {key: [] for key in data_loaders.keys()}
            for phase in data_loaders.keys():
                if phase == 'train':
                    self.encoder.train()
                    self.decoder.train()
                else:
                    self.encoder.eval()
                    self.decoder.eval()

                p_bar = tqdm(data_loaders[phase])

                for dataset in p_bar:
                    images = dataset['image'].to(self.device)
                    labels = dataset['label'].to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        predict = self.decoder(self.encoder(images))
                        loss = criterion(predict, labels)
                        iou = iou_metrics(predict, labels, ignore_id=0)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    losses[phase].append(loss.item())
                    accuracies[phase].append(iou)

                    p_bar.set_description(
                        "{}: loss={:.3f} iou={:.3f}".format(phase, np.mean(losses[phase]), np.mean(accuracies[phase]))
                    )

            self.logger.recode_score(
                (epoch + 1),
                {
                    "train_loss": np.mean(losses[list(data_loaders.keys())[0]]),
                    "train_iou": np.mean(accuracies[list(data_loaders.keys())[0]]),
                    "src_val_loss": np.mean(losses[list(data_loaders.keys())[1]]),
                    "src_val_iou": np.mean(accuracies[list(data_loaders.keys())[1]]),
                    "tgt_val_loss": np.mean(losses[list(data_loaders.keys())[2]]),
                    "tgt_val_iou": np.mean(accuracies[list(data_loaders.keys())[2]]),
                }
            )
            self.logger.save_model(
                save_best_only=False,
                encoder=self.encoder,
                decoder=self.decoder,
            )
