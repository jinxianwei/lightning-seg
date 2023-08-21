from argparse import ArgumentParser
from typing import Any, Dict, Optional

import torch
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.utils.stability import under_review
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from torchmetrics import Dice

# from src.models import UNet
from .unet import UNet


class UNetLightning(LightningModule):
    """Basic model for semantic segmentation. Uses UNet architecture by
    default.

    The default parameters in this model are for the KITTI dataset.
    Note, if you'd like to use this model as is,
    you will first need to download the KITTI dataset yourself.
    You can download the dataset `here.
    <http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015>`_

    Implemented by:

        - `Annika Brundyn <https://github.com/annikabrundyn>`_

    Example::

        from pl_bolts.models.vision import SemSegment

        model = UNetLightning(num_classes=19)
        dm = KittiDataModule(data_dir='/path/to/kitti/')

        Trainer().fit(model, datamodule=dm)

    Example CLI::

        # KITTI
        python segmentation.py --data_dir /path/to/kitti/ --accelerator=gpu
    """

    def __init__(self,
                 num_classes: int = 19,
                 num_layers: int = 5,
                 features_start: int = 64,
                 bilinear: bool = False,
                 ignore_index: Optional[int] = 19,
                 lr: float = 0.01,
                 **kwargs: Any) -> None:
        """
        Args:
            num_classes: number of output classes (default 19)
            num_layers: number of layers in each side of U-net (default 5)
            features_start: number of features in first layer (default 64)
            bilinear: whether to use bilinear interpolation (True) or
            transposed convolutions (default) for upsampling.
            ignore_index: target value to be ignored in cross_entropy
            (default 19)
            lr: learning rate (default 0.01)
        """

        super().__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear
        if ignore_index is None:
            # set ignore_index to default value of F.cross_entropy
            # if it is None.
            self.ignore_index = -100
        else:
            self.ignore_index = ignore_index
        self.lr = lr
        # self.message_hub = {}

        # self.train_dice = Dice(ignore_index=ignore_index,
        # num_classes=num_classes, average='micro')
        # 0-18的标号，共19个类别，ignore_index=19意味着为无
        # 意义区域，target中的无意义区域都标记为了19号，
        # 有意义的标号只有0-18
        self.valid_dice = Dice(average='micro', ignore_index=19)

        self.net = UNet(
            num_classes=num_classes,
            num_layers=self.num_layers,
            features_start=self.features_start,
            bilinear=self.bilinear,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def training_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Any]:
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.cross_entropy(out, mask, ignore_index=self.ignore_index)

        log_dict = {'train_loss': loss_val}
        return {'loss': loss_val, 'log': log_dict, 'progress_bar': log_dict}

    @under_review()
    def validation_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Any]:
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        # 在验证集上，模型的评估指标就不该是cross_entropy,或者可以添加其他的形式
        loss_val = F.cross_entropy(out, mask, ignore_index=self.ignore_index)

        mask_pred = torch.argmax(out, dim=1)
        self.valid_dice(mask_pred, mask)
        # self.log('valid_dice', self.valid_dice, on_step=True, on_epoch=True)
        return {'val_loss': loss_val, 'out': out}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x['val_loss'] for x in outputs]).mean()
        log_dict = {'val_loss': loss_val}
        return {
            'log': log_dict,
            'val_loss': log_dict['val_loss'],
            'progress_bar': log_dict
        }

    def configure_optimizers(self):
        # opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        # 设置sch的T_max为最大的epoch后，就会一直降低到这个学习率，
        # TODO 但我希望之后以一个学习率多训练几个epoch
        # sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        # opt, T_max=20, eta_min=1e-8)

        # TODO 去考察在模型的何处调用了这个，并进行了何种判断
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        # 5个epoch的warm_up，初始lr为0，warm_up到self.lr,
        # 而后进行cos的下降，cos最低点在20epoch处，最低lr为1e-6
        sch = LinearWarmupCosineAnnealingLR(opt,
                                            warmup_epochs=5,
                                            max_epochs=20,
                                            warmup_start_lr=0.0,
                                            eta_min=1e-6)
        return [opt], [sch]

    @staticmethod
    def add_model_specific_args(parent_parser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr',
                            type=float,
                            default=0.01,
                            help='adam: learning rate')
        parser.add_argument('--num_layers',
                            type=int,
                            default=5,
                            help='number of layers on u-net')
        parser.add_argument('--features_start',
                            type=float,
                            default=64,
                            help='number of features in first layer')
        parser.add_argument(
            '--bilinear',
            action='store_true',
            default=False,
            help='whether to use bilinear interpolation or transposed')

        return parser
