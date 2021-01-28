import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

import models
import params
from module.Preview import Preview
from module.torch.Logger import Logger


class Predict:
    def __init__(self, logger: Logger, model_path: str):
        self.logger = logger
        self.model_path = model_path
        self.preview = Preview(self.logger.log_path)
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

    @staticmethod
    def __formatting_scale__(images: np.ndarray):
        if images.dtype is not np.dtype(np.uint8):
            if np.min(images) < 0:
                images = (images - np.min(images))
            if np.max(images) <= 1:
                images = images * 255
            else:
                images = (images - np.min(images)) * (255 / np.max(images))
            images = images.astype(np.uint8)
        return images

    def create_on_mask_images(self, images, predicts, labels):
        on_mask_images = []
        print("create mask images")
        for image, predict, label in zip(tqdm(images), predicts, labels):
            image = self.__formatting_scale__(np.squeeze(image.cpu().numpy().transpose(1, 2, 0)))
            predict = predict.max(dim=0)[1].cpu().numpy()
            label = label.cpu().numpy()
            color_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # label
            for class_id in np.unique(label):
                binary_image = (label == class_id).astype(np.uint8)
                binary_image = binary_image * 255
                result = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(color_image, result[-2], -1, (0, 0, 255), 1)

            # predict
            for class_id in np.unique(predict):
                binary_image = (predict == class_id).astype(np.uint8)
                binary_image = binary_image * 255
                result = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(color_image, result[-2], -1, (255, 0, 0), 1)

            on_mask_images.append(color_image)
        return np.asarray(on_mask_images)

    def predict(self, data_loaders: list):
        for dataloader in data_loaders:
            images = []
            predicts = []
            labels = []
            with torch.no_grad():
                for dataset in dataloader:
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

                    images.extend(images_list[1])
                    predicts.extend(predict)
                    labels.extend(label)

            on_mask_images = self.create_on_mask_images(images, predicts, labels)
            for mask, predict, label in zip(on_mask_images, predicts, labels):
                self.preview.show(mask, 'mask')
                self.preview.show(predict, 'predict', type_name='torch', mode='onehot')
                self.preview.show(label, 'label', type_name='torch', mode='sparse')

