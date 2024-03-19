import torch
import time
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import math
from datetime import datetime
from PIL import Image, ImageFont, ImageDraw
from ResNet50 import resnet50


def predict(img_path):

    transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 加载图像
    img = Image.open(img_path)
    img = transform(img)
    img = img.unsqueeze(0)  # 增加一个维度作为批处理

    # 将图像送入设备 (CPU/GPU)
    img = img.to(device)

    # 进行预测
    with torch.no_grad():
         outputs = model(img)
         _, prediction = torch.max(outputs.data, dim=1)

    # 返回预测类别
    return prediction.item()

def add_label(image, label):
    # Load a font for text label
    font = ImageFont.truetype("arial.ttf", 20)  # Replace with your font path if needed

    # Create a copy of the image for labeling
    img_with_label = image.copy()

    # Draw on the copy
    draw = ImageDraw.Draw(img_with_label)
    text_bbox = draw.textbbox((0, 0), str(label), font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    draw.text((5, 5), str(label), fill=(255, 0, 0), font=font)  # Adjust text position and color as needed

    return img_with_label

def save_with_label(img_path, predicted_label, save_path):
    img = Image.open(img_path)
    labeled_img = add_label(img, predicted_label)
    filename, ext = os.path.splitext(os.path.basename(img_path))  # Separate filename and extension
    save_path = os.path.join(save_path, f"{filename}_labeled{ext}")  # Create new filename with "_labeled" suffix

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    labeled_img.save(save_path)
    # Save to same directory with same filename

if __name__ == '__main__':
    model_path = 'C:\\Users\\39080\\solarpanel\\2024_03_18Model.pth'  #模型地址
    model = resnet50()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    path = 'C:\\Users\\39080\\solarpanel\\train\\1\\'  # 预测目标文件夹
    file_list = os.listdir(path)
    predict_label = {}

    for file in file_list:
        result = predict(path + file)
        save_with_label(path + file, result, 'C:\\Users\\39080\\solarpanel\\result') #结果保存文件

        predict_label[file] = result

    print(predict_label)


#使用pandasDataFrame
# import pandas as pd
#
# predict_label = []
# for file in file_list:
#     img_path = path + file
#     result = predict(img_path)
#     predict_label.append((file, result))
#
# df = pd.DataFrame(predict_label, columns=['Filename', 'Label'])
#
# # 保存 DataFrame
# df.to_csv('predict_label.csv', index=False)