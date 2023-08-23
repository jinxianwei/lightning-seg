import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from src.callbacks import MyMonitor
from src.datamodules import KittiDataModule
from src.models import UNetLightning

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

seed_everything(1234)

# parser = ArgumentParser()
# # trainer args
# parser = Trainer.add_argparse_args(parser)
# # model args
# parser = SemSegment.add_model_specific_args(parser)
# # datamodule args
# parser = KittiDataModule.add_argparse_args(parser)

# args = parser.parse_args()

# # data
# dm = KittiDataModule(args.data_dir).from_argparse_args(args)

# # model
# model = SemSegment(**args.__dict__)

# # train
# trainer = Trainer.from_argparse_args(args)
# # print("=================================={}".format(trainer.logger))
# trainer.logger = TensorBoardLogger("tb_logs", name="kitti_seg")
# trainer.log_every_n_steps = 1
# # trainer.callbacks=[MyMonitor(),
# #                    TQDMProgressBar(),
# #                    ModelSummary(),
# #                    GradientAccumulationScheduler(),
# #                    ModelCheckpoint()]
# # print("======================={}".format(trainer.log_every_n_epoch))
# trainer.fit(model, datamodule=dm)

dm = KittiDataModule(data_dir='/home/bennie/bennie/kitti_seg/data_semantics',
                     batch_size=2)
model = UNetLightning()
logger = TensorBoardLogger('tb_logs', name='kitti_seg')
trainer = Trainer(logger=logger,
                  gpus=1,
                  min_epochs=1,
                  max_epochs=1,
                  callbacks=[MyMonitor()],
                  log_every_n_steps=1)
# trainer.fit(model, datamodule=dm)
trainer.validate(model,
                 dm,
                 ckpt_path='/home/bennie/bennie/\
lightning-seg/tb_logs/kitti_seg/version_10/checkpoints/epoch=0-step=70.ckpt')
# trainer.test(model, dm)
