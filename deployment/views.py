from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage

import math
import os
import albumentations
import numpy as np
import torch
from wtfml.data_loaders.image.classification import ClassificationDataset
from wtfml.engine import Engine
from deployment.main import CV_Model

UPLOAD_PATH = "deployment/static/image_folder/"
IMAGE_PATH = "../../static/image_folder/"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MODEL_NAME = 'se_resnext50_32x4d'
MODEL_PATH = "deployment/model/"
MODEL = CV_Model(pretrained="imagenet")
MODEL.load_state_dict(torch.load(os.path.join(MODEL_PATH, "model.bin"),map_location=torch.device('cpu')))
DEVICE = "cpu"
MODEL.to(DEVICE)
MODEL.eval()


def upload_predict(request):
    if request.method == "POST":
        file = request.FILES["image"]
        image_location = os.path.join(UPLOAD_PATH, file.name)
        image_path = os.path.join(IMAGE_PATH, file.name)
        FileSystemStorage(location=UPLOAD_PATH).save(file.name, file)        
        pred = predict(image_location, MODEL)
        pred = predict(file, MODEL)
        pred = round(sigmoid(pred)*100, 2)
        return render(request, "deployment/index.html", {"image_prediction": pred,
                                                         "image_location": image_path,
                                                         "filename": file.name}
                      )

    return render(request, "deployment/index.html", {"image_prediction": "",
                                                     "image_location": None,
                                                     "filename": None}
                  )


def predict(image_path, model):
    '''
    Make prediction using trained model
    '''

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

def sigmoid(x):
    return 1 / (1 + math.exp(-x))