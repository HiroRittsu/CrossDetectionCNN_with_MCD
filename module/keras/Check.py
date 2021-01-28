import csv
import os

import cv2
import numpy as np
from tqdm import tqdm
from keras import Model

from module.keras.CustomGenerator import CustomSequence
from module.Preview import Preview


class Check:
    def __init__(self, log_dir: str, model: Model):
        self.model = model
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.preview = Preview(self.log_dir)

    def check_all(self, generator: CustomSequence):
        data_x, data_y = generator.get()
        predicts = self.get_predicts(data_x)
        self.check_evaluate(generator)
        self.check_accuracy_per_label(data_y, predicts)
        on_mask_images = self.create_on_mask_images(data_x, predicts)
        for mask, predict, y in zip(on_mask_images, predicts, data_y):
            self.preview.show([mask, predict, y])

    def check_views(self, generator: CustomSequence):
        data_x, data_y = generator.get()
        predicts = self.get_predicts(data_x)
        on_mask_images = self.create_on_mask_images(data_x, predicts)
        for mask, predict, y in zip(on_mask_images, predicts, data_y):
            self.preview.show([mask, predict, y])

    def check_evaluate(self, generator: CustomSequence):
        with open(os.path.join(self.log_dir, "check_evaluate_score.log"), 'a') as log:
            scores = self.model.evaluate_generator(generator)
            writer = csv.writer(log)
            writer.writerow(["loss", "acc"])
            writer.writerow(scores)
            writer.writerow([])

    def check_accuracy_per_label(self, teacher_data, predict_data):
        score_list = []
        teacher_data = np.argmax(teacher_data, axis=-1)
        predict_data = np.argmax(predict_data, axis=-1)
        for y, predict in zip(teacher_data, predict_data):
            dice_scores = {}
            for label_id in np.unique(y):
                y_bool_tensor = np.where(y == label_id, True, False)
                p_bool_tensor = np.where(predict == label_id, True, False)
                common_tensor = (y_bool_tensor * p_bool_tensor)
                # dice
                dice_score = (2 * np.sum(common_tensor)) / (np.sum(y_bool_tensor) + np.sum(p_bool_tensor))
                dice_scores.setdefault(label_id, dice_score)
            score_list.append(dice_scores)

        with open(os.path.join(self.log_dir, "check_label_score.log"), 'a') as f:
            writer = csv.DictWriter(f, range(np.max(teacher_data) + 1))
            writer.writeheader()
            writer.writerows(score_list)

    def get_predicts(self, data_x):
        print("推論チェック")
        return np.asarray([self.model.predict(x[np.newaxis])[0] for x in tqdm(data_x)])

    @staticmethod
    def create_on_mask_images(image_data, predict_data):
        on_mask_images = []
        print("マスク画像生成")
        for x, predict in zip(tqdm(image_data), np.argmax(predict_data, axis=-1)):
            color_image = cv2.cvtColor(x * 255, cv2.COLOR_GRAY2BGR)
            for class_id in np.unique(predict):
                binary_image = (predict == class_id).astype(np.uint8)
                image = binary_image * 255
                _, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(color_image, contours, -1, (255, 0, 0), 1)
            on_mask_images.append(color_image)
        return np.asarray(on_mask_images)
