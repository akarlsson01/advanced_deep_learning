"""
Convolutional neural network for astronomical data with uncertainty prediction.

Also the second assignment in the Advanced Deep Learning course.

Author: Albin Karlsson
Date: 2026-03-25

Dependencies
------------
* numpy
* matplotlib
* PyTorch
* tensorboard
* alive-progress
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
            nn.Conv1d(1, 8, kernel_size=5),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(3),
            nn.Conv1d(8, 16, kernel_size=5),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(3),
            nn.Conv1d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(3),
            nn.Flatten(),
            nn.Linear(19328, 32),
            nn.ReLU(),
            # nn.Linear(302, 32),
            # nn.ReLU(),
            nn.Linear(32, 6),
        )

    def forward(self, x):
        return self.net(x)


def nllLoss(
    predictions: torch.tensor, batch_labels: torch.tensor, n_labels: int
) -> torch.tensor:
    """
    Function for the negative log likelihood.

    Parameters
    ----------
    predictions : torch.tensor
        Network predictions.
    batch_labels : torch.tensor
        The true values for the batch.
    n_labels: int
        The number of labels the network is training for.

    Returns
    -------
    The negative log-likelihood function.
    """
    mean: torch.tensor = predictions[:, :n_labels]
    log_std: torch.tensor = predictions[:, n_labels:]
    std: torch.tensor = torch.exp(log_std)
    return torch.mean((0.5 * ((batch_labels - mean) / std) ** 2) + log_std)


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
spectra: np.ndarray = np.load("../astronomy_cnn_with_pytorch/spectra.npy")
spectra_length: np.ndarray = spectra.shape[0]
# labels: mass, age, l_bol, dist, t_eff, log_g, fe_h, SNR
labelNames: list = ["mass", "age", "l_bol", "dist", "t_eff", "log_g", "fe_h", "SNR"]
labels: np.ndarray = np.load(f"../astronomy_cnn_with_pytorch/labels.npy")
# We only use the three labels: t_eff, log_g, fe_h
labelNames = labelNames[-4:-1]
labels = labels[:, -4:-1]
n_labels: np.ndarray = labels.shape[1]
n_samples: np.ndarray = labels.shape[0]
# Normalization of the data
spectra = np.log(np.maximum(spectra, 0.2))
labels_mean, labels_std = labels.mean(axis=0), labels.std(axis=0)
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
training_size: int = int(0.7 * n_samples)
validation_size: int = int(0.15 * n_samples)
test_size: int = n_samples - training_size - validation_size

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
optimizer = optim.Adam(model.parameters(), lr=1e-3)


### --- Train the model --- ###
print("\nTraining the model...\n")

#################################
num_epochs: int = 20
#################################

train_losses, val_losses = [], []
for epoch in range(num_epochs):
    with alive_bar(237) as bar:
        print("Epoch %d" % (epoch + 1))
        # Training
        model.train()
        train_loss: int = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.unsqueeze(1).to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = nllLoss(
                predictions, batch_y, n_labels
            )  # switched from MSELoss to custom nllLoss
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
                loss = nllLoss(predictions, batch_y, n_labels)
                val_loss += loss.item()
                bar()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print("Traning loss = %f, Validation loss = %f" % (train_loss, val_loss))

# Testing
model.eval()
test_loss: int = 0
true: list = []
preds: list = []
uncertainties: list = []  # separate list to store uncertainties
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.unsqueeze(1).to(device)
        batch_y = batch_y.to(device)
        predictions = model(batch_x)
        loss = nllLoss(predictions, batch_y, n_labels)
        test_loss += loss.item()
        true.append(batch_y.cpu().numpy())
        preds.append(predictions[:, :n_labels].cpu().numpy())
        uncertainties.append(
            torch.exp(predictions[:, n_labels:]).cpu().numpy()
        )  # must have torch.exp since model returns log std
test_loss /= len(test_loader)


### --- Plot the loss functions --- ###
plt.figure()
plt.plot(np.linspace(1, num_epochs, num_epochs), train_losses, label="Training loss")
plt.plot(np.linspace(1, num_epochs, num_epochs), val_losses, label="Validation loss")
ax = plt.gca()
ax.set_ylim(-1.5, 1)
plt.title("Training and validation loss functions")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("./figures/loss_functions_uncertainty")


### --- Plot True vs Prediction --- ###
for label in range(3):
    plt.figure()
    for i in range(len(true)):
        plt.scatter(true[i][:, label], preds[i][:, label], c="g", alpha=0.5)
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
    plt.savefig("./figures/performance_%s_uncertainties" % labelNames[label])


### --- Plot the pull distributions --- ###
for label in range(3):
    pull_distribution: list = []
    for i in range(len(preds)):
        pull_distribution.append(
            (preds[i][:, label] - true[i][:, label]) / uncertainties[i][:, label]
        )

    pull_distribution = np.concatenate(
        pull_distribution
    )  # This must be done in order for the sizes to function properly when plotting

    # Fit a Gaussian to the pull distribution
    mean: np.ndarray = np.mean(pull_distribution)
    std: np.ndarray = np.std(pull_distribution)

    x: np.ndarray = np.linspace(-5, 5, 200)

    plt.figure()
    plt.hist(pull_distribution, density=True, color="b", bins=50)
    plt.plot(
        x,
        1 / (std * np.sqrt(2 * np.pi)) * np.exp(-((x - mean) ** 2) / (2 * std**2)),
        color="r",
        linestyle="--",
        label=r"Fit with $\mu = $%.2f, $\sigma = $%.2f" % (mean, std),
    )
    plt.title("Pull distribution of %s" % labelNames[label])
    plt.ylabel("Counts")
    plt.xlabel(r"$(\mu - y)/\sigma$")
    plt.legend()
    plt.savefig("./figures/pull_distribution_%s" % labelNames[label])
