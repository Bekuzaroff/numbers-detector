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
            'names': ['license_plate']  # class names
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
            epochs=3,
            imgsz=640,
            batch=16,
            lr0=0.001,
            device=device
        )
        model.save("best_detector.pt")

        
        print("Обучение завершено!")
        return model
    
    def single_img_detect_number(self, im_path):
        model = YOLO("best_detector.pt") # importing trained model

        result = model(im_path)[0]
        return result