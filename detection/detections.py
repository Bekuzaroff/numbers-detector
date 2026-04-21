import os

import torch
from ultralytics import YOLO
import yaml


class Detections:
    def __init__(self):
        pass

    def create_data_yaml(self):
        """Создает конфигурационный файл для YOLO"""
        data_config = {
            'path': './',  # path to main folder of dataset
            'train': 'images/train',  # train images
            'val': 'images/val',      # validation images
            'nc': 1,                  # classes amount - 1
            'names': ['license_plate'],  # class names,
            'single_cls': True
        }
        
        with open('data.yaml', 'w') as f:
            yaml.dump(data_config, f)
    
    def fine_tune_model(self):
    # creating config for yolo
        self.create_data_yaml()
        
        # loading basic model
        model = YOLO("yolov8n.pt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)
        # Обучаем
        model.train(
            data='data.yaml',
            epochs=10,
            imgsz=640,
            batch=16,
            lr0=0.001,
            device=device
        )
        model.save("best_detector.pt")

        
        print("Обучение завершено!")
        return model
    
    def list_img_detector(self, im_folder):
        image_names = os.listdir(f"./{im_folder}/")
        model = YOLO("best_detector.pt") # importing trained model

        result = model([f'./{im_folder}/{im_name}' for im_name in image_names])
        return result