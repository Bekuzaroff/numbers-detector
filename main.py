



from preprocessing import ImagePreprocessor


if __name__ == '__main__':
    im_prep = ImagePreprocessor()

    im = im_prep.read_im("cathedral.webp")
    grey = im_prep.preprocess_image(im)
    print(grey)
    print(im)