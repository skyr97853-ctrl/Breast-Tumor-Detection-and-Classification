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
    model_s2 = resnet.resnet18()
    model_s2.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = model_s2.fc.in_features
    model_s2.fc = nn.Sequential(
        nn.Linear(num_ftrs, (int)(num_ftrs * 0.618)),
        nn.ReLU(),
        nn.Linear((int)(num_ftrs * 0.618), (int)(num_ftrs * 0.618 * 0.618)),
        nn.ReLU(),
        nn.Linear((int)(num_ftrs * 0.618 * 0.618), 4),
    )
    state_dict = torch.load("model/weights/step2.pth", weights_only=True)
    model_s2.load_state_dict(state_dict)
    model_s2.to(device)

    model_detect = YOLO('model/weights/detect.pt').to(device)

    return model_s2, model_detect


def df_maker():
    data1 = {
        'id': [],
        'image_name': [],
    }
    data2 = {
        'id': [],
        'boundary': [],
        'calcification': [],
        'direction': [],
        'shape': [],
    }
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    return df1, df2


def main_process(img_folder, output_folder, device):
    os.makedirs(output_folder, exist_ok=True)

    model_s2, model_detect = model_maker(device=device)
    df1, df2 = df_maker()

    iterator = 1
    model_s2.eval()
    with torch.no_grad():
        for filename in os.listdir(img_folder):
            img_path = os.path.join(img_folder, filename)

            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = Image.open(img_path).convert('L')
                results = model_detect.predict(source=image, conf=0, show=False, verbose=False)

                boxes = results[0].boxes.xyxy

                x1, y1, x2, y2 = map(int, boxes[0])
                img = Pre_pro.main_process(image, x1, y1, x2, y2)

                Img = torch.Tensor(np.array([[img]])).to(device)
                outputs = model_s2(Img) > 0.5
                outputs = ((outputs.int()).to('cpu')).numpy()[0]


            df1.at[iterator, 'id'] = iterator
            df1.at[iterator, 'image_name'] = os.path.splitext(filename)[0]
            df2.at[iterator, 'id'] = iterator
            df2.at[iterator, 'boundary'] = outputs[0]
            df2.at[iterator, 'calcification'] = outputs[1]
            df2.at[iterator, 'direction'] = outputs[2]
            df2.at[iterator, 'shape'] = outputs[3]
            iterator += 1

    cla_pre_path = output_folder + '/fea_pre.csv'
    cla_order_path = output_folder + '/fea_order.csv'

    df1.to_csv(cla_order_path, index=False)
    df2.to_csv(cla_pre_path, index=False)
