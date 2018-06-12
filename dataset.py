import pandas as pd
import numpy as np

import imageio
import os
import gc

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from skimage.transform import resize as imgresize

class DataSet:
    """
    """
    def __init__(self, init_path, img_shape=[224,224,3], dir_img='images_training_rev1/'):
        self.ipath = init_path
        self.img_shape = img_shape
        self.img_dir = dir_img
        self._csv_data = pd.read_csv(os.path.join(init_path,'training_solutions_rev1.csv'))


    def _get_img_path(self, filename):
        path = self.ipath+self.img_dir+str(filename)+'.jpg'
        return path

    def _load_img(self, filepath):
        return imageio.imread(filepath)

    def _load_dataset(self):
        filenames = self._csv_data.loc[:,'GalaxyID']
        self.img = []
        self.y = []
        for i, fname in tqdm(enumerate(self._csv_data['GalaxyID'].values), 'Read Img'):
            img = self._load_img(self._get_img_path(fname))
            self.img.append(imgresize(img, self.img_shape))
            self.y.append(self._csv_data.iloc[i, 1:])
        del self._csv_data; gc.collect()

    def _train_valid_split(self, random_state, train_size):
        X_train, X_valid, y_train, y_valid = train_test_split(self.img,
                                                              self.y,
                                                              random_state=random_state,
                                                              train_size=train_size,
                                                              shuffle=True)
        self.img_train = X_train
        self.img_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid
        del self.img, self.y; gc.collect()

    def load_and_transform_data(self, random_state=133, train_size=0.75):
        self._load_dataset()
        self._train_valid_split(random_state=random_state, train_size=train_size)
