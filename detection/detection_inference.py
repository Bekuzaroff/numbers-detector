


import os

import cv2 as cv
from ultralytics import YOLO

from video_processing import VideoProcessing
from detections import Detections
import torch

if __name__ == "__main__":

    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version (PyTorch): {torch.version.cuda}")

    model = YOLO("best_detector.pt") # best model
    




    
    