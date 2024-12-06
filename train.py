import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.models import resnet18  # Import model ResNet dari torchvision
from torchvision.transforms import Normalize
from Utils.getData import Data  # Pastikan Anda sudah memuat Data dengan benar
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main():
    # Hyperparameter
    BATCH_SIZE = 4
    EPOCH = 5
    LEARNING_RATE = 0.001
    NUM_CLASSES = 6

    # Pilih perangkat (CPU/GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Path ke dataset
    aug_path = "./Dataset/Augmented Images/Augmented Images/FOLDS_AUG"
    orig_path = "./Dataset/Original Images/Original Images/FOLDS"

    # Load data
    dataset = Data(base_folder_aug=aug_path, base_folder_orig=orig_path)

    # Gabungkan data train
    train_data = dataset.dataset_train + dataset.dataset_aug
    if len(train_data) == 0:
        raise ValueError("Dataset train kosong. Periksa struktur folder dataset.")

    # Buat DataLoader
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # Inisialisasi model ResNet
    model = resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")  # Menggunakan bobot ResNet pretrained
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)  # Menyesuaikan output layer dengan jumlah kelas
    model = model.to(device)  # Kirim model ke perangkat (CPU/GPU)

    # Inisialisasi loss function dan optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Training
    train_losses = train_model(train_loader, model, loss_fn, optimizer, EPOCH, device)

    # Simpan model
    torch.save(model.state_dict(), "trained_resnet18.pth")

    # Visualisasi loss
    plt.plot(range(EPOCH), train_losses, color="#3399e6", label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./training_resnet.png")
    plt.show()

def train_model(train_loader, model, loss_fn, optimizer, epochs, device):
    """
    Fungsi untuk melatih model.
    :param train_loader: DataLoader untuk data latih
    :param model: Model yang akan dilatih
    :param loss_fn: Fungsi loss
    :param optimizer: Optimizer
    :param epochs: Jumlah epoch
    :param device: Perangkat (CPU/GPU)
    :return: Daftar nilai loss per epoch
    """
    # Normalize sesuai pretrained ResNet (ImageNet mean dan std)
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_losses = []

    for epoch in range(epochs):
        model.train()
        loss_train = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (src, trg) in enumerate(train_loader):
            # Normalisasi input data
            src = src.permute(0, 3, 1, 2).float()  # Format CHW untuk PyTorch dan ubah ke float32
            src = normalize(src).to(device)  # Kirim ke perangkat
            trg = torch.argmax(trg, dim=1).long().to(device)  # Target sebagai label kategori
            
            # Forward pass
            pred = model(src)
            loss = loss_fn(pred, trg)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistik training
            loss_train += loss.item()
            _, predicted = torch.max(pred, 1)
            total_train += trg.size(0)
            correct_train += (predicted == trg).sum().item()

        accuracy_train = 100 * correct_train / total_train
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {loss_train / len(train_loader):.4f}, Accuracy: {accuracy_train:.2f}%")
        train_losses.append(loss_train / len(train_loader))

    return train_losses

if __name__ == "__main__":
    main()