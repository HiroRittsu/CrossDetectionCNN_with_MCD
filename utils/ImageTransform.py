from module.torch.SegmentationTransforms import *


class ImageTransform:
    def __init__(self, resize):
        self.data_transform = {
            'src_train': Compose([
                Resize((int(resize * 1.5), resize)),
                CenterCrop(resize),
                # RandomCrop(resize),
                # RandomRot90(),
                # Rot90(),
                # RandomGammaCorrection(),
                ToTensor(),
            ]),
            'tgt_train': Compose([
                Resize(int(resize * 1.5)),
                CenterCrop(resize),
                # RandomCrop(resize),
                # RandomRot90(),
                # RandomGammaCorrection(),
                ToTensor(),
            ]),
            'src_val': Compose([
                Resize((int(resize * 1.5), resize)),
                CenterCrop(resize),
                # Rot90(),
                ToTensor(),
            ]),
            'tgt_val': Compose([
                Resize(int(resize * 1.5)),
                CenterCrop(resize),
                ToTensor(),
            ]),
            'common': Compose([
                # RandomCrop(resize),
                # CenterCrop(resize),
                # RandomRot90(),
                ToTensor(),
            ])
        }

    def __call__(self, phase='train', **kwargs):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](kwargs)
