import os

import cv2 as cv
import numpy as np
import torch


class UnsupportedFileException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class CantOpenFileException(Exception):
    def __init__(self, *args):
        super().__init__(*args)

class Preprocessor:
    def __init__(self, target_size=(640, 640)):
        self.target_size = target_size

    def read_im(self, im_path):
        im_matr = cv.imread(im_path)
        if im_matr is None:
            raise FileNotFoundError(f"file '{im_path}' not found")
        
        allowed_format = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        ext = os.path.splitext(im_path)[1].lower()
        if ext not in allowed_format:
            raise ValueError(f"{ext} is not allowed")
        
        return im_matr

    def preprocess_image(self, im_matrix):
        # BGR to RGB
        im_matrix = cv.cvtColor(im_matrix, cv.COLOR_BGR2RGB)
        # Resize
        im_matrix = cv.resize(im_matrix, self.target_size)
        # Normalize
        im_matrix = im_matrix.astype(np.float32) / 255.0
        # Transpose (H,W,C) -> (C,H,W)
        im_matrix = im_matrix.transpose(2, 0, 1)
        # To tensor
        return torch.from_numpy(im_matrix)
    
    def read_props(self, file_path):
        bboxes = []
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 5:
                    bboxes.append([float(p) for p in parts[1:5]])
        
        if bboxes:
            return torch.tensor(bboxes, dtype=torch.float32)
        else:
            return torch.zeros((0, 4), dtype=torch.float32)
        
        

        

