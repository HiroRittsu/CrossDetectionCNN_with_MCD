import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as f
import torch.optim as optim
from tqdm import tqdm

import models
from module.torch.evaluate import iou_metrics
import params


class Adapt:
    def __init__(self, logger):
        if not os.path.exists(params.snapshot_path):
            os.makedirs(params.snapshot_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.encoder = models.Encoder(n_channels=params.input_channel).apply(models.init_weights).to(self.device)
        self.decoder1 = models.Decoder(n_classes=params.num_class).apply(models.init_weights).to(self.device)
        self.decoder2 = models.Decoder(n_classes=params.num_class).apply(models.init_weights).to(self.device)

    @staticmethod
    def discrepancy(out1, out2):
        if params.max_discrepancy:
            return torch.mean(torch.abs(f.softmax(out1, dim=1).max(dim=1)[0] - f.softmax(out2, dim=1).max(dim=1)[0]))
        else:
            return torch.mean(torch.abs(f.softmax(out1, dim=1) - f.softmax(out2, dim=1)))

    def train(self, src_train_loader, tgt_train_loader, src_test_loader, tgt_test_loader):
        cce_criterion = nn.CrossEntropyLoss()
        # symkl = Symkl2d()
        optimizer_encoder = optim.Adam(
            self.encoder.parameters(),
            lr=params.e_learning_rate,
            # betas=(params.beta1, params.beta2),
            weight_decay=5e-4
        )
        optimizer_decoder1 = optim.Adam(
            self.decoder1.parameters(),
            lr=params.c_learning_rate,
            # betas=(params.beta1, params.beta2),
            weight_decay=5e-4
        )
        optimizer_decoder2 = optim.Adam(
            self.decoder2.parameters(),
            lr=params.c_learning_rate,
            # betas=(params.beta1, params.beta2),
            weight_decay=5e-4
        )

        for epoch in range(params.adapt_num_epochs):
            self.encoder.train()
            self.decoder1.train()
            self.decoder2.train()
            p_bar = tqdm(range(params.num_iter))

            step1_c_losses = []
            step1_ious = []
            step2_c_losses = []
            step2_dis_losses = []
            step2_losses = []
            step2_ious = []
            step3_dis_losses = []
            for _ in p_bar:
                src_datasets = next(iter(src_train_loader))
                tgt_datasets = next(iter(tgt_train_loader))
                images_src_list = [
                    src_datasets['image1'].to(self.device),
                    src_datasets['image2'].to(self.device),
                    src_datasets['image3'].to(self.device)
                ]
                labels_src_list = [
                    src_datasets['label1'].to(self.device),
                    src_datasets['label2'].to(self.device),
                    src_datasets['label3'].to(self.device)
                ]
                images_tgt_list = [
                    tgt_datasets['image1'].to(self.device),
                    tgt_datasets['image2'].to(self.device),
                    tgt_datasets['image3'].to(self.device)
                ]

                with torch.enable_grad():
                    ###########################
                    #         STEP1           #
                    ###########################
                    optimizer_encoder.zero_grad()
                    optimizer_decoder1.zero_grad()
                    optimizer_decoder2.zero_grad()

                    predict_class_list = []

                    for i in range(2):
                        # データ流し込み
                        feat_src = self.encoder(images_src_list[i], images_src_list[i + 1])
                        predict_class_list.append(self.decoder1(feat_src[0], feat_src[2]))
                        predict_class_list.append(self.decoder2(feat_src[1], feat_src[2]))

                        # 誤差逆伝搬
                        loss_c1 = cce_criterion(predict_class_list[i * 2], labels_src_list[i])
                        loss_c2 = cce_criterion(predict_class_list[i * 2 + 1], labels_src_list[i + 1])
                        loss_c = loss_c1 + loss_c2
                        loss_c.backward()
                        step1_c_losses.append(loss_c.item())

                        optimizer_encoder.step()
                        optimizer_decoder1.step()
                        optimizer_decoder2.step()

                    # ログ
                    step1_ious.append(
                        iou_metrics(predict_class_list[1] + predict_class_list[2], labels_src_list[1], ignore_id=0)
                    )

                    ###########################
                    #         STEP2           #
                    ###########################
                    optimizer_encoder.zero_grad()
                    optimizer_decoder1.zero_grad()
                    optimizer_decoder2.zero_grad()

                    # データ流し込み
                    # src t-1, t
                    feat_src_1 = self.encoder(images_src_list[0], images_src_list[1])
                    predict_class2_src = self.decoder2(feat_src_1[1], feat_src_1[2])

                    # src t, t+1
                    feat_src_2 = self.encoder(images_src_list[1], images_src_list[2])
                    predict_class1_src = self.decoder1(feat_src_2[0], feat_src_2[2])

                    # tgt t-1, t
                    feat_tgt_1 = self.encoder(images_tgt_list[0], images_tgt_list[1])
                    predict_class2_tgt = self.decoder2(feat_tgt_1[1], feat_tgt_1[2])

                    # tgt t, t+1
                    feat_tgt_2 = self.encoder(images_tgt_list[1], images_tgt_list[2])
                    predict_class1_tgt = self.decoder1(feat_tgt_2[0], feat_tgt_2[2])

                    # 誤差逆伝搬
                    loss_c1 = cce_criterion(predict_class1_src, labels_src_list[1])
                    loss_c2 = cce_criterion(predict_class2_src, labels_src_list[1])
                    loss_c = loss_c1 + loss_c2
                    loss_dis = self.discrepancy(predict_class1_tgt, predict_class2_tgt)
                    loss = loss_c - loss_dis
                    loss.backward()

                    optimizer_decoder1.step()
                    optimizer_decoder2.step()

                    # ログ
                    step2_dis_losses.append(loss_dis.item())
                    step2_c_losses.append(loss_c.item())
                    step2_losses.append(loss.item())
                    step2_ious.append(
                        iou_metrics(predict_class1_src + predict_class2_src, labels_src_list[1], ignore_id=0)
                    )

                    ###########################
                    #         STEP3           #
                    ###########################
                    for _ in range(params.num_k):
                        optimizer_encoder.zero_grad()
                        optimizer_decoder1.zero_grad()
                        optimizer_decoder2.zero_grad()

                        # データ流し込み
                        # tgt t-1, t
                        feat_tgt_1 = self.encoder(images_tgt_list[0], images_tgt_list[1])
                        predict_class2 = self.decoder2(feat_tgt_1[1], feat_tgt_1[2])

                        # tgt t, t+1
                        feat_tgt_2 = self.encoder(images_tgt_list[1], images_tgt_list[2])
                        predict_class1 = self.decoder1(feat_tgt_2[0], feat_tgt_2[2])

                        # 誤差逆伝搬
                        loss_dis = self.discrepancy(predict_class1, predict_class2)
                        loss_dis.backward()
                        optimizer_encoder.step()
                        step3_dis_losses.append(loss_dis.item())

                p_bar.set_description(
                    "Epoch [{}/{}]: 1th_c_loss={:.3f} 1th_iou={:.3f} 2th_loss={:.3f} 2th_iou={:.3f} "
                    "3th_dis_loss={:.3f}".format(
                        epoch + 1,
                        params.adapt_num_epochs,
                        step1_c_losses[-1],
                        step1_ious[-1],
                        step2_losses[-1],
                        step2_ious[-1],
                        step3_dis_losses[-1],
                    )
                )

            # Predict
            tgt_losses = []
            tgt_ious = []
            src_losses = []
            src_ious = []

            self.encoder.eval()
            self.decoder1.eval()
            self.decoder2.eval()

            with torch.no_grad():
                for dataset in tgt_test_loader:
                    images_list = [
                        dataset['image1'].to(self.device),
                        dataset['image2'].to(self.device),
                        dataset['image3'].to(self.device)
                    ]
                    labels = dataset['label2'].to(self.device)

                    # tgt t-1, t
                    feat_tgt_1 = self.encoder(images_list[0], images_list[1])
                    predict2 = self.decoder2(feat_tgt_1[1], feat_tgt_1[2])

                    # tgt t, t+1
                    feat_tgt_2 = self.encoder(images_list[1], images_list[2])
                    predict1 = self.decoder1(feat_tgt_2[0], feat_tgt_2[2])

                    predict = predict1 + predict2
                    tgt_losses.append(cce_criterion(predict, labels).item())
                    tgt_ious.append(iou_metrics(predict, labels, ignore_id=0))

            with torch.no_grad():
                for dataset in src_test_loader:
                    images_list = [
                        dataset['image1'].to(self.device),
                        dataset['image2'].to(self.device),
                        dataset['image3'].to(self.device)
                    ]
                    labels = dataset['label2'].to(self.device)

                    # src t-1, t
                    feat_src_1 = self.encoder(images_list[0], images_list[1])
                    predict2 = self.decoder2(feat_src_1[1], feat_src_1[2])

                    # src t, t+1
                    feat_src_2 = self.encoder(images_list[1], images_list[2])
                    predict1 = self.decoder1(feat_src_2[0], feat_src_2[2])

                    predict = predict1 + predict2
                    src_losses.append(cce_criterion(predict, labels).item())
                    src_ious.append(iou_metrics(predict, labels, ignore_id=0))

            self.logger.recode_score(
                (epoch + 1),
                {
                    "step1_c_loss": np.mean(step1_c_losses),
                    "step1_iou": np.mean(step1_ious),
                    "step2_c_loss": np.mean(step2_c_losses),
                    "step2_dis_loss": np.mean(step2_dis_losses),
                    "step2_loss": np.mean(step2_losses),
                    "step2_iou": np.mean(step2_ious),
                    "step3_dis_loss": np.mean(step3_dis_losses),
                    "tgt_test_loss": np.mean(tgt_losses),
                    "tgt_test_iou": np.mean(tgt_ious),
                    "src_test_loss": np.mean(src_losses),
                    "src_test_iou": np.mean(src_ious),
                }
            )
            self.logger.save_model(
                save_best_only=False,
                encoder=self.encoder,
                decoder1=self.decoder1,
                decoder2=self.decoder2
            )
            print("tgt", np.mean(tgt_losses), np.mean(tgt_ious))
            print("src", np.mean(src_losses), np.mean(src_ious))
