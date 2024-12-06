import os
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, base_folder_aug, base_folder_orig):
        """
        Inisialisasi dan pemrosesan dataset.
        
        :param base_folder_aug: Path folder untuk Augmented Images
        :param base_folder_orig: Path folder untuk Original Images
        """
        self.dataset_aug = []
        self.dataset_train = []
        self.dataset_test = []
        self.dataset_valid = []

        # Memuat data dari Augmented Images (Train saja)
        self._load_data(base_folder_aug, self.dataset_aug, is_augmented=True)

        # Memuat data dari Original Images (Train, Test, Valid)
        self._load_data(base_folder_orig, self.dataset_train, data_type="Train")
        self._load_data(base_folder_orig, self.dataset_test, data_type="Test")
        self._load_data(base_folder_orig, self.dataset_valid, data_type="Valid")

        print(f"Augmented Images (Train): {len(self.dataset_aug)}")
        print(f"Original Images (Train): {len(self.dataset_train)}")
        print(f"Original Images (Test): {len(self.dataset_test)}")
        print(f"Original Images (Valid): {len(self.dataset_valid)}")

    def _load_data(self, base_folder, dataset, data_type="Train", is_augmented=False):
        """
        Memuat gambar dari folder tertentu ke dalam dataset.

        :param base_folder: Path ke folder dataset
        :param dataset: List untuk menyimpan hasil pemrosesan
        :param data_type: Jenis data (Train/Test/Valid)
        :param is_augmented: Flag untuk data augmented
        """
        onehot = np.eye(6)  # One-hot encoding untuk 6 kelas

        for fold_num in range(1, 6):
            if is_augmented:
                folder = os.path.join(base_folder, f"fold{fold_num}_AUG/Train/")
            else:
                folder = os.path.join(base_folder, f"fold{fold_num}/{data_type}/")
            
            if not os.path.exists(folder):
                continue

            for class_idx, class_name in enumerate(os.listdir(folder)):
                class_folder = os.path.join(folder, class_name)
                if not os.path.isdir(class_folder):
                    continue  # Abaikan file yang bukan direktori
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    try:
                        image = cv.resize(cv.imread(img_path), (32, 32)) / 255.0  # Normalisasi ke [0, 1]
                        dataset.append([image, onehot[class_idx]])
                    except Exception as e:
                        print(f"Error memuat gambar {img_path}: {e}")

    def __len__(self):
        """Mengembalikan jumlah total data."""
        return len(self.dataset_aug)

    def __getitem__(self, idx):
        """
        Mengembalikan tuple (image, label) dari dataset augmented.

        :param idx: Index data
        :return: Tuple (image, label) dalam format tensor
        """
        features, label = self.dataset_aug[idx]
        return (
            torch.tensor(features, dtype=torch.float32).permute(2, 0, 1),  # Convert to CHW format
            torch.tensor(label, dtype=torch.float32)
        )