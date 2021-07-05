import logging
import os
import torch
import torchvision
import torchaudio
import pytorch_lightning as pl
import hydra
from glob import glob
from omegaconf import DictConfig

from features.pcen import PCENTransform
from features.spect import Spectrogram
from GAN.DCGAN import DCGAN
from GAN.dataModule import DataModule

@hydra.main(config_path='../config', config_name='config')
def main(conf: DictConfig):
    # make features
    if conf.set.features:
        logger.info("### Feature Extraction ###")
        files = [file for path, _, _ in os.walk(conf.path.data_dir) 
                 for file in glob(os.path.join(path, '*.wav')) ]

        for file in files:
            logger.info("Processing file: {}".format(file))
            filename = file.split('/')[-1]
            class_name = file.split('/')[-2]
            filename = (
                conf.path.feat_dir +
                os.sep +
                class_name +
                os.sep +
                filename.replace('.wav', '.png')
            )

            if not os.path.isdir(conf.path.feat_dir + os.sep + class_name):
                os.makedirs(conf.path.feat_dir + os.sep + class_name)

            data, sr = torchaudio.load(file)
            feature = Spectrogram(conf=conf)(data, sr)
            #feature = PCENTransform(conf=conf)(feature)
            feature = torch.mul(
                torch.div(feature, feature.max()), 255
            )
            
            torchvision.io.write_png(
                input=torch.as_tensor(feature, dtype=torch.uint8),
                filename=filename,
                compression_level=0
            )

    if conf.set.train:
        # initiate model
        model = DCGAN(conf.hparams.beta1,
                    conf.image.img_width,
                    conf.image.img_width,
                    conf.image.img_channels,
                    conf.hparams.latent_dims,
                    conf.hparams.lr)

        logger.info("### Training ###")
        # start tensorboard logger
        dm = DataModule(conf)
        tb_logger = pl.loggers.TensorBoardLogger("logs", name="birdGAN")

        # initiate trainer
        trainer = pl.Trainer(gpus=conf.set.gpus,
			                accelerator=conf.set.accelerator,	
                            max_epochs=conf.train.epochs,
                            logger=tb_logger,
                            fast_dev_run=False)
        #start training
        trainer.fit(model, dm)

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(modules)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
