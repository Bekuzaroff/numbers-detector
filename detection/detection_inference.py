


import os

import cv2 as cv

from detections import Detections
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":

    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version (PyTorch): {torch.version.cuda}")


    detections = Detections()
   
    results = detections.list_img_detector("small_test")
        
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, (ax, result) in enumerate(zip(axes, results)):
        img_with_boxes = result.plot()
        img_rgb = cv.cvtColor(img_with_boxes, cv.COLOR_BGR2RGB)
        
        ax.imshow(img_rgb)
        ax.set_title(f"{os.path.basename(result.path)}: {len(result.boxes)} номеров")
        ax.axis('off')

    # Если фото меньше 4, остальные скрываем
    for j in range(len(results), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()