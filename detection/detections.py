import torch
from ultralytics import YOLO
import yaml


class Detections:
    def __init__(self):
        pass

    def create_data_yaml(self):
        """Создает конфигурационный файл для YOLO"""
        data_config = {
            'path': './',  # путь к корневой папке
            'train': 'images/train',  # папка с обучающими изображениями
            'val': 'images/val',      # папка с валидационными изображениями
            'nc': 1,                  # количество классов (1 - номерной знак)
            'names': ['license_plate']  # названия классов
        }
        
        with open('data.yaml', 'w') as f:
            yaml.dump(data_config, f)
    
    def fine_tune_model(self):
    # Создаем конфиг
        self.create_data_yaml()
        
        # Загружаем модель
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

        result = model(im_path) 
        return result[0].masks