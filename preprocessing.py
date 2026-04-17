import os

import cv2 as cv


class UnsupportedFileException(Exception):
    def __init__(self, *args):
        super().__init__(*args)

class ImagePreprocessor:
    def __init__(self):
        pass

    def read_im(self, im_path):        
        im_matr = cv.imread(im_path)

        if im_matr is None:
            raise FileNotFoundError(f"file '{im_path}' not found")
        

        
        allowed_format = {
                ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"
        }

        ext = os.path.splitext(im_path)[1].lower()

        if ext not in allowed_format:
                raise UnsupportedFileException(f"{ext} is not allowed")

        return im_matr

    def preprocess_image(self, im_matrix):

        # cvt to black/white
        grey_im = cv.cvtColor(im_matrix, cv.COLOR_BGR2GRAY)
        
        grey_im = grey_im.astype(float) / 255.0 # normalization

        return grey_im
        

        

