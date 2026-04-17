



from preprocessing import ImagePreprocessor


if __name__ == '__main__':
    im_prep = ImagePreprocessor()

    im = im_prep.read_im("cathedral.webp")
    print(im)