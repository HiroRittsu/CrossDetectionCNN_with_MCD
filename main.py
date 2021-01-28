from datetime import datetime
import os
import random
import sys

import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.data.dataloader import DataLoader

from core.adapt import Adapt
from core.predict import Predict
from core.pretrain import PreTrain
from core.validate import Validate
from module.torch.Logger import Logger
import params
from utils.CustomTripleRandomDataset import CustomTripleRandomDataset
from utils.CustomTripleTimeDataset import CustomTripleTimeDataset
from utils.ImageTransform import ImageTransform

cudnn.deterministic = True
np.random.seed(1337)
torch.manual_seed(1337)
random.seed(1337)


def main():
    src_train_dataset = CustomTripleRandomDataset(
        dataset_root=params.src_train_dirname,
        transform=ImageTransform(params.image_size),
        phase='src_train'
    )
    tgt_train_dataset = CustomTripleRandomDataset(
        dataset_root=params.tgt_train_dirname,
        transform=ImageTransform(params.image_size),
        phase='tgt_train'
    )
    src_val_dataset = CustomTripleTimeDataset(
        dataset_root=params.src_val_dirname,
        transform=ImageTransform(params.image_size),
        phase='src_val',
        n=1
    )
    tgt_val_dataset = CustomTripleTimeDataset(
        dataset_root=params.tgt_val_dirname,
        transform=ImageTransform(params.image_size),
        phase='tgt_val',
        n=1
    )

    if len(sys.argv) == 1:
        print('not set memo')
        memo = 'not set memo'
    else:
        memo = sys.argv[1]

    if params.pretrain:
        # load dataset
        src_train_dataloader = DataLoader(
            src_train_dataset, batch_size=params.adapt_batch_size, shuffle=True, drop_last=True
        )
        src_val_dataloader = DataLoader(
            src_val_dataset, batch_size=params.pretrain_batch_size, shuffle=False, drop_last=True
        )
        tgt_val_dataloader = DataLoader(
            tgt_val_dataset, batch_size=params.pretrain_batch_size, shuffle=False, drop_last=True
        )

        # pre train
        print('pre train')
        logger = Logger(os.path.join("log", "pretrain", datetime.now().strftime('%Y%m%d_%H%M%S')))
        logger.set_memo(memo)
        logger.set_dataset_sample([src_train_dataloader, src_val_dataloader, tgt_val_dataloader])
        PreTrain(logger).train(src_train_dataloader, src_val_dataloader, tgt_val_dataloader)

    if params.adapt:
        # load dataset
        src_train_dataloader = DataLoader(
            src_train_dataset, batch_size=params.adapt_batch_size, shuffle=True, drop_last=True
        )
        tgt_train_dataloader = DataLoader(
            tgt_train_dataset, batch_size=params.adapt_batch_size, shuffle=True, drop_last=True
        )
        src_val_dataloader = DataLoader(
            src_val_dataset, batch_size=params.adapt_batch_size, shuffle=False, drop_last=True
        )
        tgt_val_dataloader = DataLoader(
            tgt_val_dataset, batch_size=params.adapt_batch_size, shuffle=False, drop_last=True
        )

        # adapt
        print('adapt')
        logger = Logger(os.path.join("log", "adapt", datetime.now().strftime('%Y%m%d_%H%M%S')))
        logger.set_memo(memo)
        logger.set_dataset_sample([src_train_dataloader, tgt_train_dataloader, src_val_dataloader, tgt_val_dataloader])
        Adapt(logger).train(src_train_dataloader, tgt_train_dataloader, src_val_dataloader, tgt_val_dataloader)

    if params.evaluate:
        # load dataset
        src_val_dataloader = DataLoader(src_val_dataset, batch_size=1, shuffle=False)
        tgt_val_dataloader = DataLoader(tgt_val_dataset, batch_size=1, shuffle=False)

        # predict
        print('evaluate')
        logger = Logger(os.path.join("log", "evaluate", datetime.now().strftime('%Y%m%d_%H%M%S')))
        logger.set_memo(memo)
        Predict(logger, params.snapshot_path).predict([src_val_dataloader, tgt_val_dataloader])
        Validate(logger, params.snapshot_path).validate([src_val_dataloader, tgt_val_dataloader])


if __name__ == '__main__':
    main()
