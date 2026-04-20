
import os

from torch.utils.data import Dataset

from preprocessing import Preprocessor

class NumDataset(Dataset):
    def __init__(self, imgs_folder, lbls_folder, selection="train"):
        self.imgs_folder = imgs_folder
        self.lbls_folder = lbls_folder
        self.selection = selection
        self.im_names = os.listdir(f"./{imgs_folder}/{selection}/")
        self.lbl_names = os.listdir(f"./{lbls_folder}/{selection}/")
        self.preprocessor = Preprocessor()

    def __len__(self):
        return len(self.im_names)


    def __getitem__(self, index):
        im_path = f"./{self.imgs_folder}/{self.selection}/{self.im_names[index]}"
        im = self.preprocessor.read_im(im_path)
        im = self.preprocessor.preprocess_image(im)

        lbl_path = f"./{self.lbls_folder}/{self.selection}/{self.lbl_names[index]}" 
        lbl = self.preprocessor.read_props(lbl_path)

        return (im, lbl)
