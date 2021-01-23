import os
import torch

import albumentations
import pretrainedmodels

import numpy as np
import pandas as pd
import torch.nn as nn


from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.nn import functional as F

from wtfml.data_loaders.image.classification import ClassificationDataset
from wtfml.utils import EarlyStopping
from wtfml.engine import Engine

MODEL_NAME = 'se_resnext50_32x4d'
DATA_PATH = "../Dataset224/"
MODEL_PATH = "../model/"
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
DEVICE = torch.device("cuda")
EPOCHS = 1
TRAIN_BS = 32
VALID_BS = 64
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class CV_Model(nn.Module):
    def __init__(self, pretrained='imagenet'):
        super(CV_Model, self).__init__()
        self.base_model = pretrainedmodels.__dict__[MODEL_NAME](pretrained='imagenet')                
        self.out = nn.Linear(2048, 1)

    def lossfunction(self, pred, target):
        return nn.BCEWithLogitsLoss()(pred, target)

    def forward(self, image, targets):
        batch_size, _, _, _ = image.shape
        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(batch_size, -1)
        out = self.out(x)
        loss = self.lossfunction(out, targets.view(-1, 1).type_as(x))
        return out, loss


def train():
    df = pd.read_csv(DATA_PATH+"images_labeled.csv")
    X_train, X_test, y_train, y_test = train_test_split(df.image.values, 
                                                        df.label.values, 
                                                        test_size=0.3,
                                                        random_state=42,
                                                        shuffle=True)

    train_aug = albumentations.Compose(
           [albumentations.Normalize(mean=MEAN,
                                     std=STD,
                                     max_pixel_value=255.0,
                                     always_apply=True)
        ])

    valid_aug = albumentations.Compose(
           [albumentations.Normalize(mean=MEAN,
                                     std=STD,
                                     max_pixel_value=255.0,
                                     always_apply=True)
        ])

    train_images = [os.path.join(DATA_PATH, filename) for filename in X_train]
    valid_images = [os.path.join(DATA_PATH, filename) for filename in X_test]

    train_dataset = ClassificationDataset(
                                        image_paths=train_images,
                                        targets=y_train,
                                        resize=None,
                                        augmentations=train_aug,)

    train_loader = torch.utils.data.DataLoader(
                                        train_dataset,
                                        batch_size=TRAIN_BS,
                                        shuffle=True,
                                        num_workers=4)

    valid_dataset = ClassificationDataset(
                                        image_paths=valid_images,
                                        targets=y_test,
                                        resize=None,
                                        augmentations=valid_aug,)

    valid_loader = torch.utils.data.DataLoader(
                                        valid_dataset,
                                        batch_size=VALID_BS,
                                        shuffle=False,
                                        num_workers=4)

    model = CV_Model(pretrained="imagenet")
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                        optimizer,
                                        patience=3,
                                        mode="max")

    es = EarlyStopping(patience=5, mode="max")

    for epoch in range(EPOCHS):
        engine = Engine(model=model, optimizer=optimizer, device=DEVICE)
        engine.train(train_loader)
        predictions = engine.predict(valid_loader)
        predictions = np.vstack((predictions)).ravel()
        auc = metrics.roc_auc_score(y_test, predictions)
        print(f"Epoch = {epoch}, AUC = {auc}")
        scheduler.step(auc)
        es(auc, model, model_path=os.path.join(MODEL_PATH, "model.bin"))
        if es.early_stop:
            print("Early stopping")
            break      


def predict(image_path, model):


    test_aug = albumentations.Compose(
        [
            albumentations.Normalize(
                            mean=MEAN,
                            std=STD,
                            max_pixel_value=255.0,
                            always_apply=True)
        ]
    )
    test_images = [image_path]
    test_targets = [0]

    test_dataset = ClassificationDataset(
        image_paths=test_images,
        targets=test_targets,
        resize=None,
        augmentations=test_aug
        )

    test_loader = torch.utils.data.DataLoader(
                        dataset=test_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0)

    engine = Engine(model=model, optimizer=None, device=DEVICE)
    predictions = engine.predict(data_loader=test_loader)
    return np.vstack((predictions)).reshape(-1)

if __name__ == "__main__":    
    MODEL = CV_Model(pretrained="imagenet")
    MODEL.load_state_dict(torch.load(os.path.join(MODEL_PATH, "model.bin")))
    MODEL.to(DEVICE)
    MODEL.eval()

    #image_path = DATA_PATH+"/normal/1.jpg"
    image_path = DATA_PATH+"/potholes/1.jpg"
    print(predict(image_path, MODEL))
