import csv
import os
import shutil

import keras
import matplotlib.pyplot as plt
from keras.utils import plot_model


class HistoryLogger(keras.callbacks.Callback):
    def __init__(self, log_path):
        super().__init__()
        self.log_path = log_path
        self.loss_logs = {}
        self.acc_logs = {}

    def on_epoch_end(self, epoch: int, logs: dict = None):
        for key in logs.keys():
            if 'loss' in key:
                if key not in self.loss_logs:
                    self.loss_logs.setdefault(key, [])
                self.loss_logs[key].append(logs[key])
            else:
                if key not in self.acc_logs:
                    self.acc_logs.setdefault(key, [])
                self.acc_logs[key].append(logs[key])

        with open(os.path.join(self.log_path, "score.log"), 'a') as log:
            writer = csv.writer(log)
            labels = ["epoch"]
            score_list = [epoch + 1]
            labels.extend(self.loss_logs.keys())
            labels.extend(self.acc_logs.keys())
            if epoch == 0:
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


class ModelCheckpoint(keras.callbacks.ModelCheckpoint):
    def __init__(self, log_path: str, filename: str):
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        super().__init__(os.path.join(log_path, filename))


class Logger:
    def __init__(self, file_path: str, log_path: str, model: keras.Model = None):
        self.model = model
        self.log_path = log_path
        self.file_path = os.path.abspath(os.path.basename(file_path))
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        os.makedirs(os.path.join(self.log_path, "src_log"))
        shutil.copy(self.file_path, os.path.join(self.log_path, "src_log", os.path.basename(self.file_path)))
        if self.model is not None:
            self.model.summary()
            plot_model(self.model, show_shapes=True, to_file=os.path.join(self.log_path, "model.png"))

    def set_memo(self, memo: str):
        file = open(os.path.join(self.log_path, "memo.txt"), 'w')
        file.write(memo)
        file.close()

    def history_callback(self):
        return HistoryLogger(self.log_path)

    def model_full_checkpoint_callback(self):
        return ModelCheckpoint(os.path.join(self.log_path, "model_log"), "model-{epoch:02d}.hdf5")
