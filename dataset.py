import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from skimage import io
from sklearn.model_selection import train_test_split


class RGBDataset(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, nb_frames, root_dir1, root_dir2, root_dir3, annotation='./cmd_vel1.csv'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(annotation, usecols=['rosbagTimestamp', 'x', 'z']).to_numpy()
        self.data_train, _ = train_test_split(self.data, test_size=0.1, shuffle=False)
        self.nb_frames = nb_frames
        self.root_dir1 = root_dir1
        self.root_dir2 = root_dir2
        self.root_dir3 = root_dir3

    def __len__(self):
        return len(self.data_train) - 10
        # return len(self.data_train) - (self.nb_frames)
        # return 200

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X_train_t1 = [
            io.imread(os.path.join(self.root_dir1, str(int(self.data_train[idx + i, 0])) + '.jpeg')).astype(np.float32)
            for i in [0, 1, 2, 3, 4, 5]]
        X_train_t2 = [
            io.imread(os.path.join(self.root_dir2, str(int(self.data_train[idx + i, 0])) + '.jpeg')).astype(np.float32)
            for i in [0, 1, 2, 3, 4, 5]]
        X_train_t3 = [
            io.imread(os.path.join(self.root_dir3, str(int(self.data_train[idx + i, 0])) + '.jpeg')).astype(np.float32)
            for i in [0, 1, 2, 3, 4, 5]]
        X_train_t1 = np.stack(X_train_t1, axis=0)
        X_train_t2 = np.stack(X_train_t2, axis=0)
        X_train_t3 = np.stack(X_train_t3, axis=0)
        X_train = np.concatenate((X_train_t3, X_train_t2, X_train_t1), axis=2)

        X_train1 = np.moveaxis(X_train, -1, 1)

        y_train_t = [self.data_train[idx + i, 1:3].astype(np.float32) for i in [5, 6, 7, 8, 9]]
        y_train = np.stack(y_train_t, axis=0)
        y_train1 = y_train.reshape(5, 2)

        sample = {'X_train': X_train1, 'y_train': y_train1}

        return sample


class RGBDataset_test(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, nb_frames, root_dir1, root_dir2, root_dir3, annotation='./cmd_vel1.csv'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(annotation, usecols=['rosbagTimestamp', 'x', 'z']).to_numpy()
        self.data_train, self.data_test = train_test_split(self.data, test_size=0.1, shuffle=False)
        self.nb_frames = nb_frames
        self.root_dir1 = root_dir1
        self.root_dir2 = root_dir2
        self.root_dir3 = root_dir3

    def __len__(self):
        return len(self.data_test) - 10
        # return 200

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X_train_t1 = [
            io.imread(os.path.join(self.root_dir1, str(int(self.data_train[idx + i, 0])) + '.jpeg')).astype(np.float32)
            for i in [0, 1, 2, 3, 4, 5]]
        X_train_t2 = [
            io.imread(os.path.join(self.root_dir2, str(int(self.data_train[idx + i, 0])) + '.jpeg')).astype(np.float32)
            for i in [0, 1, 2, 3, 4, 5]]
        X_train_t3 = [
            io.imread(os.path.join(self.root_dir3, str(int(self.data_train[idx + i, 0])) + '.jpeg')).astype(np.float32)
            for i in [0, 1, 2, 3, 4, 5]]
        X_train_t1 = np.stack(X_train_t1, axis=0)
        X_train_t2 = np.stack(X_train_t2, axis=0)
        X_train_t3 = np.stack(X_train_t3, axis=0)
        X_train = np.concatenate((X_train_t3, X_train_t2, X_train_t1), axis=2)
        X_test1 = np.moveaxis(X_train, -1, 1)

        y_test_t = [self.data_test[idx + i, 1:3].astype(np.float32) for i in [5, 6, 7, 8, 9]]
        y_test1 = np.stack(y_test_t, axis=0)
        y_test1 = y_test1.reshape(5, 2)
        sample = {'X_test': X_test1, 'y_test': y_test1}

        return sample
