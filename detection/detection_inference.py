


import os

import cv2 as cv
from ultralytics import YOLO

from video_processing import VideoProcessing
from detections import Detections
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":

    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version (PyTorch): {torch.version.cuda}")

    model = YOLO("best_detector.pt") # best model

    # ----------- work with images
    detections = Detections()
    results = detections.list_img_detector(model, "small_test")
    detections.show_results(results)
    
    # ----------- work with video
    video_proc = VideoProcessing()
    video_path = "video.mp4"
    video_proc.track_video(model, video_path)

    # ----------- metrics
    metrics = model.val()

    
    print(f"mAP50 (при IoU=0.50): {metrics.box.map50:.4f}")
    print(f"mAP50-95 (средний при IoU 0.5:0.95): {metrics.box.map:.4f}") 
    print(f"Precision (Точность): {metrics.box.mp:.4f}") 
    print(f"Recall (Полнота): {metrics.box.mr:.4f}")   




    
    