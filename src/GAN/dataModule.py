import torch
import pytorch_lightning as pl
from torchvision import transforms as T
from torchvision import datasets

class DataModule(pl.LightningDataModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.feat_dir = self.conf.path.feat_dir
        self.img_width = self.conf.image.img_width

    def setup(self, stage=None):
        transform = T.Compose([T.Grayscale(num_output_channels=1),
                               T.Resize((self.img_width, self.img_width)),
                               T.CenterCrop(self.img_width),
                               T.RandomHorizontalFlip(p=0.5),
                               T.ToTensor()])
        self.train_data = datasets.ImageFolder(self.feat_dir, transform=transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data,
                                           batch_size=self.conf.train.batch_size,
                                           shuffle=True,
                                           num_workers=self.conf.set.num_workers)