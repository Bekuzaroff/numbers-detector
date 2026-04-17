import os

import cv2 as cv


class UnsupportedFileException(Exception):
    def __init__(self, *args):
        super().__init__(*args)

    def __str__(self):
        return "unsupported file format, please, " \
        "choose another file type"

class ImagePreprocessor:
    def __init__(self):
        pass

    def read_im(self, im_path):
        try:
            allowed_format = {
                ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"
            }

            format = os.path.splitext(im_path)[1].lower()

            if format not in allowed_format:
                raise UnsupportedFileException
        
            im_matr = cv.imread(im_path)


            if im_matr is None:
                raise FileNotFoundError

            return im_matr
        
        except FileNotFoundError as e:

            return "file not found"
        
        except UnsupportedFileException as e:

            return e.__str__()
        

        

