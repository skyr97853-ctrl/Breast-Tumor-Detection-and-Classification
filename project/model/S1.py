import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
import PIL.Image as Image
from .ResNet import resnet as resnet
from ultralytics import YOLO
from . import Pre_pro


def model_maker(device):
    model_s1 = resnet.resnet18()
    model_s1.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = model_s1.fc.in_features
    model_s1.fc = nn.Sequential(
        nn.Linear(num_ftrs, (int)(num_ftrs * 0.618)),
        nn.ReLU(),
        nn.Linear((int)(num_ftrs * 0.618), (int)(num_ftrs * 0.618 * 0.618)),
        nn.ReLU(),
        nn.Linear((int)(num_ftrs * 0.618 * 0.618), 6),
    )
    state_dict = torch.load("model/weights/step1.pth", weights_only=True)
    model_s1.load_state_dict(state_dict)
    model_s1.to(device)

    model_detect = YOLO('model/weights/detect.pt').to(device)

    return model_s1, model_detect


def df_maker():
    data1 = {
        'id': [],
        'images_name': [],
    }
    data2 = {
        'id': [],
        'label': [],
    }
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    return df1, df2


def main_process(img_folder, output_folder, device):
    os.makedirs(output_folder, exist_ok=True)

    model_s1, model_detect = model_maker(device=device)
    df1, df2 = df_maker()

    iterator = 1
    model_s1.eval()
    with torch.no_grad():
        for filename in os.listdir(img_folder):
            img_path = os.path.join(img_folder, filename)

            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = Image.open(img_path).convert('L')
                results = model_detect.predict(source=image, conf=0, show=False, verbose=False)

                boxes = results[0].boxes.xyxy

                x1, y1, x2, y2 = map(int, boxes[0])
                img = Pre_pro.main_process(image, x1, y1, x2, y2)

                Img = torch.Tensor(np.array([[img]])).to('cuda')
                outputs = model_s1(Img)
                _, predicted = torch.max(outputs, 1)
                prediction = predicted.item()

            df1.at[iterator, 'id'] = iterator
            df1.at[iterator, 'images_name'] = os.path.splitext(filename)[0]
            df2.at[iterator, 'id'] = iterator
            df2.at[iterator, 'label'] = prediction + 1
            iterator += 1

    cla_pre_path = output_folder + '/cla_pre.csv'
    cla_order_path = output_folder + '/cla_order.csv'

    df1.to_csv(cla_order_path, index=False)
    df2.to_csv(cla_pre_path, index=False)
