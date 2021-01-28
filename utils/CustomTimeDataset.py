import os
import re
import shutil
from datetime import datetime
from glob import glob

import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image
from tqdm import tqdm


class CustomTimeDataset(data.Dataset):
    def __init__(
            self,
            source_dataset_root: str,
            target_dataset_root: str,
            transform=None,
            source_phase='train',
            target_phase='train',
            common_phase='common'
    ):
        self.source_dataset_root = source_dataset_root
        self.target_dataset_root = target_dataset_root
        self.transform = transform
        self.source_phase = source_phase
        self.target_phase = target_phase
        self.common_phase = common_phase
        self.tmp_dir = os.path.join('.tmp', datetime.now().strftime('%H%M%S%f'))
        self.fp_src_set = {'image': None, 'label': None}
        self.fp_tgt_set = {'image': None, 'label': None}
        self.data_size = 0

        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        # source
        images, labels = self.load(self.source_dataset_root)

        # データを移し替え
        # image
        filename = os.path.join(self.tmp_dir, 'src_image')
        self.fp_src_set['image'] = np.memmap(filename, dtype=np.uint8, mode='w+', shape=images.shape)
        self.fp_src_set['image'][:] = images[:]
        # label
        filename = os.path.join(self.tmp_dir, 'src_label')
        self.fp_src_set['label'] = np.memmap(filename, dtype=np.int, mode='w+', shape=labels.shape)
        self.fp_src_set['label'][:] = labels[:]

        # target
        images, labels = self.load(self.target_dataset_root)

        # データを移し替え
        # image
        filename = os.path.join(self.tmp_dir, 'tgt_image')
        self.fp_tgt_set['image'] = np.memmap(filename, dtype=np.uint8, mode='w+', shape=images.shape)
        self.fp_tgt_set['image'][:] = images[:]
        # label
        filename = os.path.join(self.tmp_dir, 'tgt_label')
        self.fp_tgt_set['label'] = np.memmap(filename, dtype=np.int, mode='w+', shape=labels.shape)
        self.fp_tgt_set['label'][:] = labels[:]

    def __del__(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree('.tmp')

    def load(self, dataset_root):
        # ディレクトリに含まれる画像の読み取り。
        # 画像、ラベルデータのファイル名に含まれる数字をもとにソートする。
        dataset_files = [
            sorted(
                glob(os.path.join(dirname, "*.*")), key=self.__numerical_sort__
            ) for dirname in glob(os.path.join(dataset_root, "**/"))
        ]

        assert len(dataset_files) != 0, "データセットが見つかりません : " + dataset_root

        image_filenames = []
        label_filenames = []
        # PILライブラリで画像を試し読みして、modeで画像orラベルを判断する。
        for i in range(2):
            if Image.open(dataset_files[i][0]).mode == "P":
                image_filenames = dataset_files[1 - i]
                label_filenames = dataset_files[i]
        assert bool(len(image_filenames)), "Error: label image mode is must 'P'"
        assert len(image_filenames) == len(label_filenames), "Error: the dataset is must the same sample size"
        self.data_size = max(len(image_filenames), self.data_size)

        images = []
        labels = []
        for (image_filename, label_filename) in tqdm(zip(image_filenames, label_filenames), total=len(image_filenames)):
            images.append(cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE))
            labels.append(np.asarray(Image.open(label_filename)))

        return np.asarray(images)[:, :, :, np.newaxis], np.asarray(labels)

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
        src_index = index * len(self.fp_src_set['image']) // self.data_size
        src_image = self.fp_src_set['image'][src_index]
        src_label = self.fp_src_set['label'][src_index]
        tgt_index = index * len(self.fp_tgt_set['image']) // self.data_size
        tgt_image = self.fp_tgt_set['image'][tgt_index]
        tgt_label = self.fp_tgt_set['label'][tgt_index]

        src_dataset = self.transform(self.source_phase, image=src_image, label=src_label)
        tgt_dataset = self.transform(self.target_phase, image=tgt_image, label=tgt_label)

        return self.transform(
            self.common_phase,
            src_image=src_dataset['image'],
            src_label=src_dataset['label'],
            tgt_image=tgt_dataset['image'],
            tgt_label=tgt_dataset['label']
        )
