import os
import pickle
import random
import re
import shutil
from abc import abstractmethod
from datetime import datetime
from glob import glob

import cv2
import numpy as np
from PIL import Image
from keras.utils import Sequence, np_utils


class CustomSequence(Sequence):
    """
    KerasのSequenceを継承するクラス。
    デフォルトのメソッドに以外に、データセットの逐次保存をメインとした機能が実装されている。
    """

    @abstractmethod
    def generate_dataset(self, start_index, end_index):
        """
        __getitem__関数内で呼び出す。
        返り値はバッチのx,y（tuple型）である必要がある。
        :param start_index:
        :param end_index:
        :return: x,y
        """
        pass

    def __init__(self,
                 x_shape: tuple = None,
                 y_shape: tuple = None,
                 batch_size: int = None,
                 rescale: float = None,
                 x_type=None,
                 y_type=None,
                 cache_dir: str = None,
                 is_generation: bool = True):
        """
        初期化。
        memmap関数でデータセットの保存用に保存領域を確保する。
        なお、cache_dirに指定がなければ.cacheという名前のディレクトリが自動で生成される。
        :param x_shape:
        :param y_shape:
        :param batch_size:
        :param rescale:
        :param x_type:
        :param y_type:
        """
        self.rescale = rescale
        self.can_dump_reading = True
        self.cache_dir = cache_dir
        self.is_generation = is_generation
        self.entity_cache_dir = self.cache_dir if self.cache_dir is not None \
            else os.path.join('.cache', datetime.now().strftime('%H%M%S%f'))
        self.x_dump_filepath = os.path.join(self.entity_cache_dir, "x.dump")
        self.y_dump_filepath = os.path.join(self.entity_cache_dir, "y.dump")
        self.info_dump_filepath = os.path.join(self.entity_cache_dir, "info.dump")
        self.ready_indexes = []

        # キャッシュディレクトリの作成
        if not os.path.exists(self.entity_cache_dir):
            os.makedirs(self.entity_cache_dir)

        # info dumpファイルがない場合
        if not os.path.exists(self.info_dump_filepath):
            assert x_type is not None and x_shape is not None, "x_type, x_shape指定必須"
            assert y_type is not None and y_shape is not None, "y_type, y_shape指定必須"
            with open(self.info_dump_filepath, mode='wb') as f:
                # x_shapeなどの情報保存
                pickle.dump([x_type, x_shape, y_type, y_shape, batch_size], f)
                # データセット保存用の領域を確保（ファイルとして実体化される）
                np.memmap(
                    self.x_dump_filepath,
                    dtype=x_type,
                    mode='w+',
                    shape=x_shape
                )
                np.memmap(
                    self.y_dump_filepath,
                    dtype=y_type,
                    mode='w+',
                    shape=y_shape
                )
                self.can_dump_reading = False

        # 保存情報の読み取り
        with open(self.info_dump_filepath, mode='rb') as f:
            info = pickle.load(f)
            self.x_type = info[0]
            self.x_shape = info[1]
            self.y_type = info[2]
            self.y_shape = info[3]
            self.batch_size = info[4]

        # 実体化されたダンプファイルを、読み込み書き込みモードで準備
        self.x_fp = np.memmap(
            self.x_dump_filepath,
            dtype=self.x_type,
            mode='r+',
            shape=self.x_shape
        )
        self.y_fp = np.memmap(
            self.y_dump_filepath,
            dtype=self.y_type,
            mode='r+',
            shape=self.y_shape
        )

        print("ダンプファイル読み取り:", self.can_dump_reading)

    def __del__(self):
        """
        後処理。
        もしcache_pathが指定されてなければ.cache/を削除
        :return: None
        """
        if self.cache_dir is None:
            if os.path.exists(".cache/"):  # ディレクトリがあれば
                shutil.rmtree(".cache/")
        del self.x_fp, self.y_fp

    def __getitem__(self, index):
        """
        実装必須関数。Kerasのfit_generatorなどの内部から呼び出される。
        返り値はバッチのx,yで返す必要がある。
        :param index:
        :return: x, y
        """
        start_index = index * self.batch_size
        end_index = start_index + self.batch_size
        if not self.can_dump_reading or self.is_generation:
            x, y = self.generate_dataset(start_index, end_index)
            self.x_fp[start_index:end_index] = x
            self.y_fp[start_index:end_index] = y
        return np.asarray(self.x_fp[start_index:end_index]), np.asarray(self.y_fp[start_index:end_index])

    def __len__(self):
        """
        実装必須関数。Kerasのfit_generatorなどの内部から呼び出される。
        返り値は(サンプル数/バッチサイズ)で整数とする必要がある。
        :return:
        """
        return self.x_shape[0] // self.batch_size

    def on_epoch_end(self):
        """
        毎エポック終了時に呼び出される関数。
        バッチサイズによって呼び出されないデータがあるため、ここで端数のデータを格納する。
        :return: None
        """
        # 端数を格納
        if not self.can_dump_reading and not self.x_shape[0] % self.batch_size == 0:
            start_index = self.x_shape[0] - (self.x_shape[0] % self.batch_size)
            end_index = self.x_shape[0]
            x, y = self.generate_dataset(start_index, end_index)
            self.x_fp[start_index:end_index] = x
            self.y_fp[start_index:end_index] = y
        self.can_dump_reading = True

    def get(self, size: int = -1):
        """
        内部から呼び出される関数とは別に、データセットをサンプルとして読み出したい時に利用することを想定。
        引数に個数を指定する。
        なお、size = -1の場合は全て取り出す。
        :param size:
        :return: x,y
        """
        x_data = []
        y_data = []
        max_size = self.__len__() * self.batch_size
        data_size = max_size if size == -1 else size
        # 重複なし
        if size <= max_size:
            selected_indexes = set()
            while len(x_data) < data_size:
                if len(self.ready_indexes) == 0:
                    self.ready_indexes = random.sample(range(self.__len__()), self.__len__())
                select_index = self.ready_indexes.pop()
                if select_index in selected_indexes:
                    continue
                selected_indexes.add(select_index)
                dataset = self.__getitem__(select_index)
                for x, y in zip(dataset[0], dataset[1]):
                    x_data.append(x)
                    y_data.append(y)
        # 重複あり
        else:
            while len(x_data) < data_size:
                if len(self.ready_indexes) == 0:
                    self.ready_indexes = random.sample(range(self.__len__()), self.__len__())
                dataset = self.__getitem__(self.ready_indexes.pop())
                for x, y in zip(dataset[0], dataset[1]):
                    x_data.append(x)
                    y_data.append(y)

        return np.asarray(x_data[:size]), np.asarray(y_data[:size])

    def get_batch(self):
        """
        バッチサイズのデータ取得専用関数
        :return:
        """
        if len(self.ready_indexes) == 0:
            self.ready_indexes = random.sample(range(self.__len__()), self.__len__())
        return self.__getitem__(self.ready_indexes.pop())


class CustomImageDataGeneratorFlow(CustomSequence):
    """
    すでに変数に格納されているデータを逐次読み出すクラス。
    （Kerasですでに同じ機能を持つメソッドがあるため、このクラスを使うメリットはあまりなかったりする）
    """

    def __init__(self,
                 x_data: np.ndarray,
                 y_data: np.ndarray,
                 cache_dir: str = None,
                 batch_size: int = 32,
                 x_type=np.float32,
                 y_type=np.uint8,
                 x_shape=None,
                 y_shape=None,
                 num_classes: int = 0,
                 rescale: float = None):
        """
        初期化。
        今のところは画像データという想定でデータ整形をする。
        なお、effect_functionはまだ実装していない。
        :param x_data:
        :param y_data:
        :param batch_size:
        :param x_type:
        :param y_type:
        :param num_classes:
        :param rescale:
        """
        assert 2 < len(x_data.shape) < 5, \
            "Error: x.shape -> (samples, height, width) or (samples, height, width, channel)"
        assert 0 < len(y_data.shape) < 3, "Error: y.shape -> (samples) or (samples, class)"
        assert x_data.shape[0] == y_data.shape[0], "Error: samples are not same size"

        self.x_data = x_data.astype(x_type)
        self.y_data = y_data.astype(y_type)

        # グレースケールの画像の場合
        # if len(self.x_data.shape) == 3:
        #    self.x_data = self.x_data.reshape((self.x_data.shape[0], self.x_data.shape[1], self.x_data.shape[2], 1))
        # スケールの変更
        if rescale is not None:
            self.x_data = self.x_data * rescale
        # one-hotベクトル型でない場合
        if len(self.y_data.shape) == 1:
            self.y_data = np_utils.to_categorical(
                self.y_data,
                num_classes=num_classes if bool(num_classes) else self.y_data.max() + 1
            )

        # CustomSequenceの初期化。データセット保存に必要な情報を渡す
        super().__init__(
            self.x_data.shape if x_shape is None else x_shape,
            self.y_data.shape if y_shape is None else y_shape,
            batch_size,
            rescale,
            x_type,
            y_type,
            cache_dir,
            False
        )

    @abstractmethod
    def data_effect(self, x_data: np.ndarray, y_data: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        画像処理用関数。
        :param x_data:
        :param y_data:
        :return: effected_x_data, effected_y_data
        """
        return x_data, y_data

    def generate_dataset(self, start_index, end_index):
        """
        batch_indexからの情報をもとにデータの取り出しを行う関数。
        :param start_index:
        :param end_index:
        :return: x.y
        """
        x = self.x_data[start_index:end_index]
        y = self.y_data[start_index:end_index]
        # 画像、ラベルデータの加工
        x, y = self.data_effect(x, y)
        return np.asarray(x), np.asarray(y)


class CustomCategoryImageDataGeneratorFromDirectory(CustomSequence):
    """
    * 指定されたディレクトリから自動でラベル付けし、データセット化する関数。
    * Kerasにも同様のメソッドが実装されているが、それと比べて更に省メモリ化、高速化されている（ﾊｽﾞ）。
    * directoryで指定された情報をもとにデータセットの逐次保存をするダンプファイルの名前を決定している。
    * そのため、同じdirectoryが指定されている場合は、新たにデータセットを作成するのではなく、
    * 保存されているダンプファイルからデータを取り出すようにしている。
    """

    def __init__(self,
                 directory: str = None,
                 cache_dir: str = None,
                 target_size: tuple = None,
                 rescale: float = None,
                 read_mode: str = 'color',
                 num_classes: int = 0,
                 batch_size: int = None,
                 x_type=None,
                 y_type=None,
                 class_mode: str = "binary",
                 shuffle: bool = True,
                 interpolation: int = cv2.INTER_LINEAR,
                 x_effect_function=None):
        """
        初期化。
        指定されたディレクトリから画像を読み取る。
        それと同時に、それぞれの画像が格納されているディレクトリの名前をラベルとして画像を自動でラベリングする。
        なお、x_effect_functionに関数オブジェクトを指定することで、画像を加工することが可能。
        x_effect_functionは、引数は画像一枚で返り値も画像一枚という条件を満たす必要がある。
        :param directory:
        :param target_size:
        :param rescale:
        :param read_mode:
        :param num_classes:
        :param batch_size:
        :param x_type:
        :param y_type:
        :param class_mode:
        :param shuffle:
        :param interpolation:
        :param x_effect_function:
        """
        assert directory is not None or cache_dir is not None, "directory もしくは cache_dirを指定する必要があります"

        self.directory = directory
        self.target_size = target_size
        self.read_mode = read_mode
        self.interpolation = interpolation
        self.x_effect_function = x_effect_function
        self.y_label_names = None
        self.x_filename = []
        self.y_data = None

        x_shape = None
        if self.directory is not None:
            self.y_label_names = glob(os.path.join(self.directory, "**/"))  # 画像の検索
            x = []
            y = []
            # 見つかった画像とその画像が入っているディレクトリの名前をリストに格納
            for i, label_name in enumerate(self.y_label_names):
                x_filenames = glob(os.path.join(label_name, "*.*"))
                x.extend(x_filenames)
                y.extend([i] * len(x_filenames))

            # シャッフル
            if shuffle:
                dataset = list(zip(x, y))
                np.random.shuffle(dataset)
                x, y = zip(*dataset)
            self.x_filename = list(x)
            self.y_data = np.asarray(y)

            # categoryの場合はone-hotベクトルに整形
            if class_mode == 'category':
                self.y_data = np_utils.to_categorical(
                    self.y_data,
                    num_classes=num_classes if bool(num_classes) else self.y_data.max() + 1
                )
            # binaryの場合は(sample数,1)の形に整形
            else:
                self.y_data = self.y_data.reshape((-1, 1))
            num_channel = 1 if self.read_mode == 'gray' else 3
            x_shape = (len(self.x_filename), self.target_size[0], self.target_size[1], num_channel)

        # CustomSequenceの初期化。データセット保存に必要な情報を渡す
        super().__init__(
            x_shape,
            self.y_shape,
            batch_size,
            rescale,
            x_type,
            y_type,
            cache_dir
        )

    def generate_dataset(self, start_index, end_index):
        """
        batch_indexの情報をもとに画像ファイルを読み取り、データセットとして返す関数。
        :param start_index:
        :param end_index:
        :return: x,y
        """
        x_data = []
        for i, filename in enumerate(self.x_filename[start_index:end_index]):
            # カラーで読み取る場合
            if self.read_mode == 'color':
                # 画像読み込み
                image = cv2.imread(filename, cv2.IMREAD_COLOR)
                # 画像の加工
                if self.x_effect_function is not None:
                    image = self.x_effect_function(image)
                # サイズの変更
                if self.target_size is not None:
                    if not self.target_size[0] == image.shape[0] and not self.target_size[1] == image.shape[1]:
                        image = cv2.resize(image, self.target_size, interpolation=self.interpolation)
                # OpenCVで読み取るため、カラーチャンネルの反転処理
                image = image[:, :, [2, 1, 0]]
            # グレースケールで読みよる場合
            else:
                # 画像読み込み
                image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                # 画像の加工
                if self.x_effect_function is not None:
                    image = self.x_effect_function(image)
                # サイズの変更
                if self.target_size is not None:
                    if not self.target_size[0] == image.shape[0] and not self.target_size[1] == image.shape[1]:
                        image = cv2.resize(image, self.target_size, interpolation=self.interpolation)
                # チャンネルの追加
                image = image[:, :, np.newaxis]
            x_data.append(image)
        x = np.asarray(x_data, dtype=self.x_type)
        # スケールの変更
        x = x if self.rescale is None else x * self.rescale
        # 画像に対応するラベルの読み取り
        y = np.asarray(self.y_data[start_index:end_index], dtype=self.y_type)
        return x, y


class CustomSegmenticalImageDataGeneratorFromDirectory(CustomSequence):
    """
    * 指定されたディレクトリから目標画像とラベル画像を判別し、データセット化する関数。
    * directoryで指定された情報をもとにデータセットの逐次保存をするダンプファイルの名前を決定している。
    * そのため、同じdirectoryが指定されている場合は、新たにデータセットを作成するのではなく、
    * 保存されているダンプファイルからデータを取り出すようにしている。
    * なお、ラベル画像はカラーパレット情報を含むものでなければならない。
    * 画像とラベルの対応付けは、数字が一致していれば自動で処理するようにしている。
    """

    def __init__(self,
                 directory: str = None,
                 cache_dir: str = None,
                 num_classes: int = None,
                 target_size: tuple = None,
                 rescale: float = None,
                 read_mode: str = 'color',
                 batch_size: int = None,
                 x_type=None,
                 y_type=None,
                 class_mode: str = 'category',
                 shuffle: bool = True,
                 interpolation: int = cv2.INTER_LINEAR,
                 num_samples: int = 0):
        """
        初期化。
        指定されたディレクトリから画像を読み取る。
        そして、カラーパレットを含む画像をラベルデータとして自動でデータセットとして整形する。
        data_effect関数は、入力されるデータの前処理を行うことができる。
        なお、is_generationを有効にすると重複ありランダムファイル参照をするため、data_effect関数で乱数を元にした画像処理を
        することで同じ画像ファイルが生成されないような処理ができる。
        :param directory:
        :param num_classes:
        :param cache_dir:
        :param target_size:
        :param rescale:
        :param read_mode:
        :param batch_size:
        :param x_type:
        :param y_type:
        :param class_mode:
        :param shuffle:
        :param interpolation:
        """
        assert directory is not None or cache_dir is not None, "directory もしくは cache_dirを指定する必要があります"

        self.directory = directory
        self.target_size = target_size
        self.read_mode = read_mode
        self.num_classes = num_classes
        self.class_mode = class_mode
        self.interpolation = interpolation
        self.x_filenames = []
        self.y_filenames = []

        x_shape = None
        y_shape = None
        if self.directory is not None:
            # ディレクトリに含まれる画像の読み取り。
            # 画像、ラベルデータのファイル名に含まれる数字をもとにソートする。
            image_file_set = [
                sorted(
                    glob(os.path.join(dirname, "*.*")), key=self.numerical_sort
                ) for dirname in glob(os.path.join(self.directory, "**/"))
            ]

            x = []
            y = []
            # PILライブラリで画像を読み取り、modeで画像orラベルを判断する。
            for i in range(2):
                if Image.open(image_file_set[i][0]).mode == "P":
                    x = image_file_set[1 - i]
                    y = image_file_set[i]
            assert bool(len(x)), "Error: label image mode is must 'P'"
            assert len(x) == len(y), "Error: the dataset is must the same sample size"

            # シャッフル
            if shuffle:
                dataset = list(zip(x, y))
                np.random.shuffle(dataset)
                x, y = zip(*dataset)
            self.x_filenames = list(x)
            self.y_filenames = list(y)

            num_samples = len(self.x_filenames) if num_samples == 0 else num_samples
            num_channel = 1 if self.read_mode == 'gray' else 3
            x_shape = (num_samples, self.target_size[0], self.target_size[1], num_channel)
            y_shape = (num_samples, self.target_size[0], self.target_size[1], self.num_classes)

        # CustomSequenceの初期化。データセット保存に必要な情報を渡す
        super().__init__(
            x_shape,
            y_shape,
            batch_size,
            rescale,
            x_type,
            y_type,
            cache_dir,
            bool(num_samples > len(self.x_filenames))
        )

    @abstractmethod
    def data_effect(self, x_data: np.ndarray, y_data: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        画像処理用関数。
        :param x_data:
        :param y_data:
        :return: effected_x_data, effected_y_data
        """
        return x_data, y_data

    def generate_dataset(self, start_index, end_index):
        """
        batch_indexの情報をもとに画像ファイルを読み取り、データセットとして返す関数。
        :param start_index:
        :param end_index:
        :return: x,y
        """
        x_data = []
        y_data = []
        # 逐次生成の場合
        if self.is_generation:
            target_indexes = random.choices(range(len(self.x_filenames)), k=end_index - start_index)
            target_x_filenames = np.asarray(self.x_filenames)[target_indexes]
            target_y_filenames = np.asarray(self.y_filenames)[target_indexes]
        # 再生成なしの場合
        else:
            target_x_filenames = self.x_filenames[start_index:end_index]
            target_y_filenames = self.y_filenames[start_index:end_index]

        # 画像、ラベルデータの読み取り
        for x_filename, y_filename in zip(target_x_filenames, target_y_filenames):
            # ラベルデータの読み取り
            # PILライブラリを利用して読み取り
            y_image = np.asarray(Image.open(y_filename))

            # 画像データの読み取り
            # カラーの場合
            if self.read_mode == 'color':
                # 画像を読み取り
                x_image = cv2.imread(x_filename, cv2.IMREAD_COLOR)
                # 画像、ラベルデータの加工
                x_image, y_image = self.data_effect(x_image, y_image)
                # 画像サイズの変更
                if self.target_size is not None:
                    if not self.target_size[0] == x_image.shape[0] and not self.target_size[1] == x_image.shape[1]:
                        x_image = cv2.resize(x_image, self.target_size, interpolation=self.interpolation)
                # OpenCVで読み取るため、カラーチャンネルの反転処理
                x_image = x_image[:, :, [2, 1, 0]]
            # グレースケールの場合
            else:
                # 画像を読み取り
                x_image = cv2.imread(x_filename, cv2.IMREAD_GRAYSCALE)
                # 画像、ラベルデータの加工
                x_image, y_image = self.data_effect(x_image, y_image)
                # 画像サイズの変更
                if self.target_size is not None:
                    if not self.target_size[0] == x_image.shape[0] and not self.target_size[1] == x_image.shape[1]:
                        x_image = cv2.resize(x_image, self.target_size, interpolation=self.interpolation)
                # チャンネルの追加
                x_image = x_image[:, :, np.newaxis]

            # categoryの場合はラベルをone-hotベクトルに整形
            if self.class_mode == 'category':
                y_image = np.eye(self.num_classes)[y_image.astype(self.y_type)]

            x_data.append(x_image)
            y_data.append(y_image)

        x = np.asarray(x_data, dtype=self.x_type)
        y = np.asarray(y_data, dtype=self.y_type)
        # 画像のスケール変更
        x = x if self.rescale is None else x * self.rescale
        return x, y

    @staticmethod
    def numerical_sort(value):
        """
        ファイル名の数字を元にソートする関数
        :param value:
        :return:
        """
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts
