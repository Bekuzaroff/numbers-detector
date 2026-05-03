

import torch
from ultralytics import YOLO
import yaml

from detection.detections import Detections





if __name__ == '__main__':
    detection = Detections()
    model = detection.fine_tune_model()
    
    # Simple test
    results = model("cathedral.webp")
    results[0].show()

    print(detection.single_img_detect_number("cathedral.webp"))