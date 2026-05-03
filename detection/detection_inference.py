


import os

import cv2 as cv
from ultralytics import YOLO

from video_processing import VideoProcessing
from detections import Detections
import torch
import random as rand
import easyocr

if __name__ == "__main__":

    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version (PyTorch): {torch.version.cuda}")

    model = YOLO("best_detector.pt") # best model
    detection = Detections()
    img_names = os.listdir("small_test/")
    results = model([os.path.join("small_test/", name) for name in img_names])
    numbers = []

    for i, result in enumerate(results):

        (mat_img, numbers) = detection.label_img(result)
        print(numbers)
        break


        

        

    

    
    




    
    