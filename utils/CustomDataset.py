﻿import os
import re
import shutil
from datetime import datetime
from glob import glob

import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image
from tqdm import tqdm


class CustomDataset(data.Dataset):
    def __init__(self, dataset_root: str, transform=None, phase='train'):
        self.dataset_root = dataset_root
        self.transform = transform
        self.phase = phase
        self.tmp_dir = os.path.join('.tmp', datetime.now().strftime('%H%M%S%f'))
        self.fp_set = {'image': None, 'label': None}
        self.data_size = None

        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        # ディレクトリに含まれる画像の読み取り。
        # 画像、ラベルデータのファイル名に含まれる数字をもとにソートする。
        dataset_files = [
            sorted(
                glob(os.path.join(dirname, "*.*")), key=self.__numerical_sort__
            ) for dirname in glob(os.path.join(self.dataset_root, "**/"))
        ]

        assert len(dataset_files) != 0, "データセットが見つかりません : " + self.dataset_root

        image_filenames = []
        label_filenames = []
        # PILライブラリで画像を試し読みして、modeで画像orラベルを判断する。
        for i in range(2):
            if Image.open(dataset_files[i][0]).mode == "P":
                image_filenames = dataset_files[1 - i]
                label_filenames = dataset_files[i]
        assert bool(len(image_filenames)), "Error: label image mode is must 'P'"
        assert len(image_filenames) == len(label_filenames), "Error: the dataset is must the same sample size"
        self.data_size = len(image_filenames)

        images = []
        labels = []
        for (image_filename, label_filename) in tqdm(zip(image_filenames, label_filenames), total=len(image_filenames)):
            images.append(cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE))
            labels.append(np.asarray(Image.open(label_filename)))

        # データを移し替え
        # image
        images = np.asarray(images)[:, :, :, np.newaxis]
        filename = os.path.join(self.tmp_dir, 'image')
        self.fp_set['image'] = np.memmap(filename, dtype=np.uint8, mode='w+', shape=images.shape)
        self.fp_set['image'][:] = images[:]
        # label
        labels = np.asarray(labels)
        filename = os.path.join(self.tmp_dir, 'label')
        self.fp_set['label'] = np.memmap(filename, dtype=np.int, mode='w+', shape=labels.shape)
        self.fp_set['label'][:] = labels[:]

    def __del__(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree('.tmp')

    @staticmethod
    def __numerical_sort__(value):
        """
        ファイル名の数字を元にソートする関数
        :param value:
        :return:
        """
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        image = self.fp_set['image'][index]
        label = self.fp_set['label'][index]
        return self.transform(self.phase, image=image, label=label)
