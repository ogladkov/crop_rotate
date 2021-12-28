import os
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import gdown


class CFG:
    data_dir = './data'
    input_dir = './input'
    tst_path = './test'
    out_path = './out'
    cropped_path = 'cropped'
    num_classes = 2
    img_height = 512
    img_width = 512
    img_size = (img_height, img_width)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = 'se_resnext50_32x4d'
    encoder_weights = 'imagenet'
    activation = 'sigmoid'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    # testimgs_dir = './input'
    # output_dir = './output'
    # grass_img_path = 'grass.jpg'
    weights_path = 'best_model.pth'
    # weights_download_link = "https://drive.google.com/uc?id=1xwDw5K3leyqvUFxb5__tm4WKCUtTX8RZ"
    # grass_download_link = "https://drive.google.com/uc?id=1KspEkqQ4aWI3J5zJvRG51Iivx0zx8SDs"
    model = torch.load(weights_path, map_location=device).to(device)


def to_tensor(x, **kwargs):
    return x.astype('float32').transpose(2, 0, 1)


def infer(cfg, img_path):
    # Infer model. Returns original image and predicted mask
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data_transforms = {
        "test": A.Compose([
        A.Resize(*cfg.img_size),
        A.Lambda(image=cfg.preprocessing_fn),
        # A.Lambda(image=to_tensor),
        ToTensorV2()
    ], p=1.0)
    }
    transformer = data_transforms['test']
    img_aug = transformer(image=img)
    img_aug = img_aug['image']
    # print(img_aug.shape)
    img_aug = img_aug.expand(1, -1, -1, -1)
    img_aug = img_aug.to(cfg.device).float()
    prediction = cfg.model(img_aug)
    prediction = prediction.detach().cpu().numpy()
    prediction = np.round(prediction, 0).astype('uint8') * 255
    msk = prediction[0, 1, :, :]
    msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=0)
    return img, msk


# cfg = CFG()

