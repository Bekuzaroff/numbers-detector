import os
import cv2 as cv
import numpy as np
import torch
from ultralytics import YOLO
import ultralytics
import yaml
import random as rand

class Detections:
    def __init__(self):
        pass

    def create_data_yaml(self):
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
    
    def label_img(self, result) -> list[np.array, list[np.array]]:

        if not isinstance(result, ultralytics.engine.results.Results):
            raise TypeError(f"wrong type of result returned by model")
        
        
        labeled_img = result.orig_img.copy()
        boxes = result.boxes
        cut_nums = []

        for box in boxes:

            coords = box.xyxy
            x1, y1, x2, y2 = list(map(int, coords.tolist()[0]))
            

            if coords.numel() != 0:

                labeled_img = cv.rectangle(labeled_img, (x1, y1), (x2, y2),  (rand.randint(0, 255), 
                                                                        rand.randint(0, 255), 
                                                                        rand.randint(0, 255)), thickness=2)
                cut_nums.append(labeled_img[y1:y2, x1:x2])
            
        
        return (labeled_img, cut_nums)
    