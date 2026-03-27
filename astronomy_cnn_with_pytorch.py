"""
Convolutional neural network for astronomical data.

Also the first assignment in the Advanced Deep Learning course.

Author: Albin Karlsson
Date: 2026-03-25
"""

### --- Importing relevant packages --- ###
import matplotlib.pyplot as plt
import numpy as np

# from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from alive_progress import alive_bar
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.LazyLinear(3),
        )

    def forward(self, x):
        return self.net(x)


### --- Download the data --- ###
# hf_hub_download(
#     repo_id="simbaswe/galah4", filename="labels.npy", repo_type="dataset", local_dir="."
# )
# hf_hub_download(
#     repo_id="simbaswe/galah4",
#     filename="spectra.npy",
#     repo_type="dataset",
#     local_dir=".",
# )


### --- Set GPU as device --- ###
device: torch.device = torch.device("mps")


### --- Load the data --- ###
# This code is copied from the assignment page
spectra: np.ndarray = np.load("./spectra.npy")
spectra_length: np.ndarray = spectra.shape[0]
# labels: mass, age, l_bol, dist, t_eff, log_g, fe_h, SNR
labelNames: list = ["mass", "age", "l_bol", "dist", "t_eff", "log_g", "fe_h", "SNR"]
labels: np.ndarray = np.load(f"./labels.npy")
# We only use the three labels: t_eff, log_g, fe_h
labelNames = labelNames[-4:-1]
labels = labels[:, -4:-1]
n_labels: np.ndarray = labels.shape[0]
# Normalization of the data
spectra = np.log(np.maximum(spectra, 0.2))
labels_mean, labels_std = labels.mean(), labels.std()
labels = (labels - labels_mean) / labels_std
# labels /= np.max(labels, axis=0)


### --- Convert data to tensors --- ###
spectra_tensor: torch.tensor = torch.tensor(spectra, dtype=torch.float32).to(device)
labels_tensor: torch.tensor = torch.tensor(labels, dtype=torch.float32).to(device)


### --- Visualize a few spectra --- ###
# example_spectra: int = 5
# print("Producing plots of %d spectra" % example_spectra)
# with alive_bar(example_spectra) as bar:
#     for i in range(example_spectra):
#         fig, ax = plt.subplots(1, 1, figsize=(10, 5))
#         ax.plot(spectra[i], lw=1)
#         ax.set_title("Star %d" % i)
#         ax.set_xlabel("Flux measurement number")
#         ax.set_ylabel("Normalized wave length (log scale)")
#         plt.tight_layout()
#         plt.savefig("./figures/star_%d" % i)
#         bar()


### --- Split the data into training, validation and test --- ###
training_size: int = int(0.7 * n_labels)
validation_size: int = int(0.15 * n_labels)
test_size: int = n_labels - training_size - validation_size

train_data, val_data, test_data = random_split(
    TensorDataset(spectra_tensor, labels_tensor),
    [training_size, validation_size, test_size],
)


### --- Create dataloaders --- ###
batch_size: int = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


### --- Initialize model --- ###
model = CNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


### --- Train the model --- ###
print("\nTraining the model\n")
num_epochs: int = 10
train_losses, val_losses = [], []
for epoch in range(num_epochs):
    with alive_bar(279) as bar:
        print("Epoch %d" % (epoch + 1))
        # Training
        model.train()
        train_loss: int = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.unsqueeze(1).to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            bar()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss: int = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.unsqueeze(1).to(device)
                batch_y = batch_y.to(device)
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
                bar()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Testing
        model.eval()
        test_loss: int = 0
        true = []
        preds = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.unsqueeze(1).to(device)
                batch_y = batch_y.to(device)
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                test_loss += loss.item()
                true.append(batch_y.cpu().numpy())
                preds.append(predictions.cpu().numpy())
                bar()
        test_loss /= len(test_loader)


### --- Plot the loss functions --- ###
plt.figure()
plt.plot(np.linspace(1, num_epochs, num_epochs), train_losses, label="Training loss")
plt.plot(np.linspace(1, num_epochs, num_epochs), val_losses, label="Validation loss")
ax = plt.gca()
ax.set_ylim(0, 1)
plt.title("Training and validation loss functions")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("./figures/loss_functions")


### --- Plot True vs Prediction --- ###
for label in range(3):
    plt.figure()
    for i in range(len(true)):
        plt.scatter(true[i][:, label], preds[i][:, label], c="g")
    plt.plot(
        [0, 1],
        [0, 1],
        transform=ax.transAxes,
        linestyle="--",
        color="k",
        label="Perfect prediction",
    )
    plt.title("Model performance for %s" % labelNames[label])
    plt.xlabel("True value")
    plt.ylabel("Predicted value")
    plt.savefig("./figures/performance_%s" % labelNames[label])
