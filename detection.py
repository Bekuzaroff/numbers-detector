import os
from ultralytics import YOLO

from preprocessing import Preprocessor

# the class with all detections stuff (model, training etc.)

class Detections:
    def __init__(self):
        self.model = YOLO("yolov8n.pt") # our main detection model
        self.preprocessor = Preprocessor() # custom preprocessing class

    def fine_tune(self, imgs_path, anns_path):
        img_names = os.listdir(imgs_path) # only names "image.jpg"
        props_names = os.listdir(anns_path) # "image.txt"
        return
            

        

        



        


