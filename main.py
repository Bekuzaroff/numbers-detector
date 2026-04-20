

import torch
from ultralytics import YOLO
import yaml

def create_data_yaml():
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

def fine_tune_model():
    # Создаем конфиг
    create_data_yaml()
    
    # Загружаем модель
    model = YOLO("yolov8n.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    # Обучаем
    results = model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        lr0=0.001,
        device=device
    )
    
    print("Обучение завершено!")
    return model

if __name__ == '__main__':
    model = fine_tune_model()
    
    # Тестируем на изображении
    results = model("cathedral.webp")
    results.show()