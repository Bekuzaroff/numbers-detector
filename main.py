



from detection import Detections
from preprocessing import Preprocessor


if __name__ == '__main__':
    im_prep = Preprocessor()

    im = im_prep.read_im("cathedral.webp")
    grey = im_prep.preprocess_image(im)
    print(grey)
    print(im)

    det = Detections()

    det.fine_tune("./images/", "./labels/")