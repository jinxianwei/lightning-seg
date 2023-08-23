import pytorch_lightning as pl
# import 的等级必须是models和nn
import torch.nn as nn
import torchvision.models as models


class Classification_2d(pl.LightningModule):

    def __init__(self, label_dict={}, log_dir=''):
        super().__init__()
        self.num_classes = len(label_dict)
        self.net = models.resnet18(pretrained=True)
        # resnet 系列
        self.fc = nn.Linear(self.net.fc.in_features, self.num_classes)
        self.net.fc = nn.Identity()

        self.loss = nn.L1Loss
        self.label_dict = label_dict
        self.label_to_name_dict = {v: k for k, v in label_dict.items()}

        self.training_save = True
        self.log_dir = log_dir
