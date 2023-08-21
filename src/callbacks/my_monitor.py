from typing import Any

import numpy as np
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT

classes = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole'
           'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky',
           'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
           'bicycle')

palette = {
    0: (128, 64, 128),
    1: (244, 35, 232),
    2: (70, 70, 70),
    3: (102, 102, 156),
    4: (190, 153, 153),
    5: (153, 153, 153),
    6: (250, 170, 30),
    7: (220, 220, 0),
    8: (107, 142, 35),
    9: (152, 251, 152),
    10: (70, 130, 180),
    11: (220, 20, 60),
    12: (255, 0, 0),
    13: (0, 0, 142),
    14: (0, 0, 70),
    15: (0, 60, 100),
    16: (0, 80, 100),
    17: (0, 0, 230),
    18: (119, 11, 32)
}


def draw_sem_seg(image, sem_seg, classes, palette):
    # 这里的image和classes 预留为图片和mask图像混合的接口
    sem_seg = sem_seg.cpu().data
    ids = np.unique(sem_seg)[::-1]
    # legal_indices = ids < num_classes
    labels = np.array(ids, dtype=np.int64)
    # 对于ignore的idx，将其gt的mask设置为黑色[0, 0, 0]
    colors = [
        palette[label] if label in palette else [0, 0, 0] for label in labels
    ]
    mask = np.zeros((sem_seg.shape[0], sem_seg.shape[1], 3), dtype=np.uint8)

    for label, color in zip(labels, colors):
        mask[sem_seg == label, :] = color
    mask = mask.transpose(2, 0, 1)
    return mask


class MyMonitor(Callback):
    """Weight update rule from Bootstrap Your Own Latent (BYOL).

    Updates the target_network params using an exponential moving average
    update rule weighted by tau.
    BYOL claims this keeps the online_network from collapsing.

    The PyTorch Lightning module being trained should have:

        - ``self.online_network``
        - ``self.target_network``

    .. note:: Automatically increases tau from ``initial_tau`` to 1.0
    with every training step

    Args:
        initial_tau (float, optional): starting tau. Auto-updates
        with every training step

    Example::

        # model must have 2 attributes
        model = Model()
        model.online_network = ...
        model.target_network = ...

        trainer = Trainer(callbacks=[BYOLMAWeightUpdate()])
    """

    def __init__(self) -> None:
        super().__init__()

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                           outputs: STEP_OUTPUT, batch: Any,
                           batch_idx: int) -> None:
        pl_module.log('train_loss',
                      outputs['loss'],
                      on_step=True,
                      on_epoch=True)
        # pl_module.log("lr", pl_module.lr)
        lr = pl_module.optimizers().param_groups[0]['lr']
        pl_module.log('lr', lr, on_step=True, on_epoch=False)
        return super().on_train_batch_end(trainer, pl_module, outputs, batch,
                                          batch_idx)

    def on_validation_batch_end(self, trainer: Trainer,
                                pl_module: LightningModule,
                                outputs: STEP_OUTPUT, batch: Any,
                                batch_idx: int, dataloader_idx: int) -> None:
        img, mask = batch
        out = outputs['out']
        mask_pred = torch.argmax(out, dim=1)
        seg_color = draw_sem_seg(img[0], mask[0], classes, palette)
        mask_pred_color = draw_sem_seg(img[0], mask_pred[0], classes, palette)

        pl_module.logger.experiment.add_image(
            'kitti_images', img[0], pl_module.global_step + batch_idx)
        pl_module.logger.experiment.add_image(
            'kitti_mask', seg_color, pl_module.global_step + batch_idx)
        pl_module.logger.experiment.add_image(
            'pred_mask', mask_pred_color, pl_module.global_step + batch_idx)
        pl_module.log('valid_dice',
                      pl_module.valid_dice,
                      on_step=True,
                      on_epoch=True)

        return super().on_validation_batch_end(trainer, pl_module, outputs,
                                               batch, batch_idx,
                                               dataloader_idx)
