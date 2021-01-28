import csv
import os
import zipfile
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch

from module.Preview import Preview


class Logger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.preview = Preview(log_path, 'sampling')
        self.save_src_path = os.path.join(self.log_path, "src.zip")
        self.model_path = os.path.join(self.log_path, "models.zip")
        self.score_path = os.path.join(self.log_path, "score.log")
        self.memo_path = os.path.join(self.log_path, "memo.txt")
        self.save_model_count = 0
        self.best = None
        self.monitor_logs = None
        self.monitor_op = None
        self.loss_logs = {}
        self.acc_logs = {}
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.save_src_files()

    def save_src_files(self):
        with zipfile.ZipFile(self.save_src_path, 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
            for filename in glob('**/*.py', recursive=True):
                new_zip.write(filename)

    def set_memo(self, memo: str):
        file = open(self.memo_path, 'w')
        file.write(memo)
        file.close()

    def save_model(self, mode='min', monitor='val_loss', save_best_only=True, tmp_filename='.tmp.pt', **kwargs):
        if self.best is None and save_best_only:
            if mode == 'min':
                self.best = np.Inf
                self.monitor_op = np.less
            else:
                self.best = -np.Inf
                self.monitor_op = np.greater

            if 'loss' in monitor:
                self.monitor_logs = self.loss_logs[monitor]
            else:
                self.monitor_logs = self.acc_logs[monitor]

        if save_best_only:
            current = self.monitor_logs[-1]
            if self.monitor_op(current, self.best):
                with zipfile.ZipFile(self.model_path, 'w', compression=zipfile.ZIP_DEFLATED) as model_zip:
                    self.best = current
                    for key in kwargs.keys():
                        torch.save(kwargs[key].state_dict(), tmp_filename)
                        model_zip.write(tmp_filename, arcname='{}.pt'.format(key))
                os.remove(tmp_filename)
        else:
            with zipfile.ZipFile(self.model_path, 'a', compression=zipfile.ZIP_DEFLATED) as model_zip:
                self.save_model_count += 1
                for key in kwargs.keys():
                    torch.save(kwargs[key].state_dict(), tmp_filename)
                    model_zip.write(tmp_filename, arcname='{}-{}.pt'.format(key, self.save_model_count))
            os.remove(tmp_filename)

    def recode_score(self, epoch: int, logs: dict = None):
        for key in logs.keys():
            if 'loss' in key:
                if key not in self.loss_logs:
                    self.loss_logs.setdefault(key, [])
                self.loss_logs[key].append(logs[key])
            else:
                if key not in self.acc_logs:
                    self.acc_logs.setdefault(key, [])
                self.acc_logs[key].append(logs[key])

        with open(self.score_path, 'a') as log:
            writer = csv.writer(log)
            labels = ["epoch"]
            score_list = [epoch]
            labels.extend(self.loss_logs.keys())
            labels.extend(self.acc_logs.keys())
            if os.stat(self.score_path).st_size == 0:
                writer.writerow(labels)
            for key in labels[1:]:
                score_list.append(logs[key])
            writer.writerow(score_list)

        for key, value in self.loss_logs.items():
            plt.plot(value, label=key, marker=".")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.log_path, "loss_history.png"))
        plt.clf()

        for key, value in self.acc_logs.items():
            plt.plot(value, label=key, marker=".")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.log_path, "acc_history.png"))
        plt.clf()

    def set_dataset_sample(self, data_loaders: list):
        for data_loader in data_loaders:
            data = next(iter(data_loader))
            for key in data.keys():
                if 'label' in key:
                    self.preview.show(data[key], type_name='torch', mode='sparse', title=key)
                elif 'image' in key:
                    self.preview.show(data[key], type_name='torch', title=key)
