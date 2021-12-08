import os
import torch
import cv2
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp


class CFG:
    data_dir = './data'
    work_dir = '.'
    tst_path = './test'
    out_path = './out'
    num_classes = 2
    img_height = 512
    img_width = 512
    img_size = (img_height, img_width)
    batch_size = 4
    device = 'cuda'
    epochs = 40
    classes = ['photos', 'bg']
    encoder = 'se_resnext50_32x4d'
    encoder_weights = 'imagenet'
    activation = 'sigmoid'

cfg = CFG()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transforms=None):
        self.img_paths = sorted([os.path.join(cfg.tst_path, i) for i in os.listdir(cfg.tst_path)])
        self.transforms = transforms
        self.classes = cfg.classes
        self.class_values = [self.classes.index(cls.lower()) for cls in self.classes]

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            data = self.transforms(image=img)
            img = data['image']

        return img

    def __len__(self):
        return len(self.img_paths)

preprocessing_fn = smp.encoders.get_preprocessing_fn(cfg.encoder, cfg.encoder_weights)

def to_tensor(x, **kwargs):
    return x.astype('float32').transpose(2, 0, 1)

data_transforms = {
    "test": A.Compose([
        A.Resize(*cfg.img_size),
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor),
    ], p=1.0)
}

test_dataset = Dataset(cfg, transforms=data_transforms['test'])
test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size,
                         num_workers=2, shuffle=True, pin_memory=True, drop_last=False)

model = smp.Unet(
    encoder_name=cfg.encoder,
    encoder_weights=cfg.encoder_weights,
    classes=len(cfg.classes),
    activation=cfg.activation,
)

def write_masks():
    cntr = 0
    if not os.path.exists(cfg.out_path):
        os.mkdir(cfg.out_path)
    model = torch.load('./best_model.pth', cfg.device)
    for test_batch in iter(test_loader):
        test_batch = test_batch.float().to(cfg.device)
        with torch.no_grad():
            predicted_masks = model(test_batch).detach().cpu().numpy().transpose(0, 2, 3, 1)
        test_batch = test_batch.to('cpu').numpy().transpose(0, 2, 3, 1)
        for x in range(cfg.batch_size):
            mask = predicted_masks[x][:, :, 1]
            mask = np.round(mask, 0) * 255
            out_fname = os.path.join(cfg.out_path, f'{cntr}.jpg')
            cv2.imwrite(out_fname, mask)
            cntr += 1

write_masks()