import argparse
import io
import os
import sys
import time
from datetime import datetime

import awkward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from gnn_encoder import GNNEncoder, collate_fn_gnn
# from gnn_trafo_helper import (
#     denormalize_x,
#     denormalize_y,
#     evaluate_model,
#     get_img_from_matplotlib,
#     normalize_time,
#     normalize_x,
#     normalize_y,
#     train_model,
# )
from matplotlib import pyplot as plt
from rich.progress import Progress
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from torch_geometric.data import Batch, Data
from torch_geometric.nn import DynamicEdgeConv, global_mean_pool, knn_graph

DATA_PATH = "./"  # path to the data

# Load the dataset
train_dataset = awkward.from_parquet(os.path.join(DATA_PATH, "train.pq"))
val_dataset = awkward.from_parquet(os.path.join(DATA_PATH, "val.pq"))
test_dataset = awkward.from_parquet(os.path.join(DATA_PATH, "test.pq"))

# to get familiar with the dataset, let's inspect it.
print(f"The training dataset contains {len(train_dataset)} events.")
print(f"The validation dataset contains {len(val_dataset)} events.")
print(f"The test dataset contains {len(test_dataset)} events.")
print(f"The training dataset has the following columns: {train_dataset.fields}")
print(f"The validation dataset has the following columns: {val_dataset.fields}")
print(f"The test dataset has the following columns: {test_dataset.fields}")
# print the first event of the training dataset
print(f"The first event of the training dataset is: {train_dataset[0]}")

# We are interested in the labels xpos and ypos. This is the position of the neutrino interaction that we want to predict.
print(
    f"The first event of the training dataset has the following labels: {train_dataset['xpos'][0]}, {train_dataset['ypos'][0]}"
)
# Awkward arrays also allow us to obtain the 'xpos' and 'ypos' label for all events in the dataset
print(
    f"The first 10 labels of the training dataset are: {train_dataset['xpos'][:10]}, {train_dataset['ypos'][:10]}"
)

# The data can be accessed by using the 'data' key.
# The data is a 3D array with the first dimension being the number of events,
# the second dimension being the the three features (time, x, y)
# the third dimension being the number of hits,
print(
    f"The first event of the training dataset has {len(train_dataset['data'][0][0])} hits, i.e., detected photons."
)
# Let's loop over all hits and print the time, x, and y coordinates of the first event.
for i in range(len(train_dataset["data"][0, 0])):
    print(
        f"Hit {i}: time = {train_dataset['data'][0,0,i]}, x = {train_dataset['data'][0,1, i]}, y = {train_dataset['data'][0,2,i]}"
    )
# To get all hit times of the first event, you can use the following code:
print(
    f"The first event of the training dataset has the following hit times: {train_dataset['data'][0, 0]}"
)
print(
    f"The first event of the training dataset has the following hit x positions: {train_dataset['data'][0, 1]}"
)
print(
    f"The first event of the training dataset has the following hit y positions: {train_dataset['data'][0, 2]}"
)

times_mean = np.mean(train_dataset["data"][:, 0:1, :])
times_std = np.std(train_dataset["data"][:, 0:1, :])
x_mean = np.mean(train_dataset["data"][:, 1:2, :])
y_mean = np.mean(train_dataset["data"][:, 2:3, :])
x_std = np.std(train_dataset["data"][:, 1:2, :])
y_std = np.std(train_dataset["data"][:, 2:3, :])


def normalize_dataset(dataset) -> None:
    """
    Function to normalize the datasets.
    """
    # Normalize data and labels
    # working with Awkward arrays is a bit tricky because the ['data'] field can't be assigned in-place,
    # so we need to extract the time, x, and y coordinates, normalize them separately,
    # and then concatenate them back together.
    times = dataset["data"][
        :, 0:1, :
    ]  # important to index the time dimension with 0:1 to keep this dimension (n_events, 1, n_hits)
    # with [:,0,:] we would get a 2D array of shape (n_events, n_hits)
    norm_times = (times - times_mean) / times_std
    x = dataset["data"][:, 1:2, :]
    norm_x = (x - x_mean) / x_std
    y = dataset["data"][:, 2:3, :]
    norm_y = (y - y_mean) / y_std

    # Concatenate the normalized data back together
    dataset["data"] = awkward.concatenate([norm_times, norm_x, norm_y], axis=1)
    # Normalize labels (this can be done in-place), e.g. by
    dataset["xpos"] = (dataset["xpos"] - x_mean) / x_std
    dataset["ypos"] = (dataset["ypos"] - y_mean) / y_std


normalize_dataset(train_dataset)
normalize_dataset(val_dataset)
normalize_dataset(test_dataset)

# Hint: You can define a helper function to normalize the data and you can use the same normalization for the validation and test datasets.


# Create the DataLoader for training, validation, and test datasets
# Important: We use the custom collate function to preprocess the data for GNN (see the description of the collate function for details)
def collate_fn_gnn(batch):
    """
    Custom function that defines how batches are formed.

    For a more complicated dataset with variable length per event and Graph Neural Networks,
    we need to define a custom collate function which is passed to the DataLoader.
    The default collate function in PyTorch Geometric is not suitable for this case.

    This function takes the Awkward arrays, converts them to PyTorch tensors,
    and then creates a PyTorch Geometric Data object for each event in the batch.

    You do not need to change this function.

    Parameters
    ----------
    batch : list
        A list of dictionaries containing the data and labels for each graph.
        The data is available in the "data" key and the labels are in the "xpos" and "ypos" keys.
    Returns
    -------
    packed_data : Batch
        A batch of graph data objects.
    labels : torch.Tensor
        A tensor containing the labels for each graph.
    """
    data_list = []
    labels = []

    for b in batch:
        # this is a loop over each event within the batch
        # b["data"] is the first entry in the batch with dimensions (n_features, n_hits)
        # where the feautures are (time, x, y)
        # for training a GNN, we need the graph notes, i.e., the individual hits, as the first dimension,
        # so we need to transpose to get (n_hits, n_features)
        tensordata = torch.from_numpy(b["data"].to_numpy()).T
        # the original data is in double precision (float64), for our case single precision is sufficient
        # we let's convert to single precision (float32) to save memory and computation time
        tensordata = tensordata.to(dtype=torch.float32)

        # PyTorch Geometric needs the data in a specific format
        # we need to create a PyTorch Geometric Data object for each event
        this_graph_item = Data(x=tensordata)
        data_list.append(this_graph_item)

        # also the labels need to be packaged as pytorch tensors
        labels.append(torch.Tensor([b["xpos"], b["ypos"]]).unsqueeze(0))

    labels = torch.cat(labels, dim=0)  # convert the list of tensors to a single tensor
    packed_data = Batch.from_data_list(
        data_list
    )  # convert the list of Data objects to a single Batch object
    return packed_data, labels


### --- Important parameters --- ###
if __name__ == "__main__":
    print("\nGraph Neural Network\n")
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", default=16, help="Batch size in int.")
    parser.add_argument(
        "-lr", "--learning_rate", default=1e-4, help="Learning rate in float."
    )
    parser.add_argument(
        "-e", "--epochs", default=2, help="Set the number of training epochs"
    )  # set low for trial training as default
    parser.add_argument(
        "-k", default=5, help="Set the kernel size"
    )  # set low for trial training as default
    args = parser.parse_args()
    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)
    epochs = int(args.epochs)
    k = int(args.k)
    print("Batch size: %d" % batch_size)
    print("Learning rate: %f" % learning_rate)
    print("Epochs: %d\n" % epochs)


train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_gnn
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_gnn
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_gnn
)


# Defintion of the GNN model
# Use the DynamicEdgeConv layer from the pytorch geometric package like this:
# MLP is a Multi-Layer Perceptron that is used to compute the edge features, you still need to define it.
# The input dimension to the MLP should be twice the number of features in the input data (i.e., 2 * n_features),
# because the edge features are computed from the concatenation of the two nodes that are connected by the edge.
# The output dimension of the MLP is the new feauture dimension of this graph layer.
from torch_geometric.nn import DynamicEdgeConv


class MLP(nn.Module):
    def __init__(self, input, output):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input, output * 2),
            nn.ReLU(),
            nn.Linear(output * 2, output),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class GNNEncoder(nn.Module):
    def __init__(self):
        super(GNNEncoder, self).__init__()
        self.layer_list = nn.ModuleList(
            [
                DynamicEdgeConv(
                    MLP(2 * 3, 32),
                    aggr="mean",
                    k=k,  # k is the number of nearest neighbors to consider
                ),
                DynamicEdgeConv(
                    MLP(2 * 32, 128),
                    aggr="mean",
                    k=k,  # k is the number of nearest neighbors to consider
                ),
                DynamicEdgeConv(
                    MLP(2 * 128, 256),
                    aggr="mean",
                    k=k,  # k is the number of nearest neighbors to consider
                ),
                DynamicEdgeConv(
                    MLP(2 * 256, 64),
                    aggr="mean",
                    k=k,  # k is the number of nearest neighbors to consider
                ),
                DynamicEdgeConv(
                    MLP(2 * 64, 32),
                    aggr="mean",
                    k=k,  # k is the number of nearest neighbors to consider
                ),
                DynamicEdgeConv(
                    MLP(2 * 32, 16),
                    aggr="mean",
                    k=k,  # k is the number of nearest neighbors to consider
                ),
            ]
        )

        self.final_mlp = nn.Sequential(nn.Linear(16, 2))

    def forward(self, data):
        # data is a batch graph item. it contains a list of tensors (x) and how the batch is structured along this list (batch)
        x = data.x
        batch = data.batch

        # loop over the DynamicEdgeConv layers:
        for layer in self.layer_list:
            x = layer(x, batch)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        # the output of the last layer has dimensions (n_batch, n_nodes, graph_feature_dimension)
        # where n_batch is the number of graphs in the batch and n_nodes is the number of nodes in the graph
        # i.e. one output per node (i.e. the hits in the event).
        # To combine all node feauters into single prediction, we recommend to use global pooling
        x = global_mean_pool(x, batch)  # -> (n_batch, output_dim)
        # x is now a tensor of shape (n_batch, output_dim)

        # either your the last graph feature dimension is already the output dimension you want to predict
        # or you need to add a final MLP layer to map the output dimension to the number of labels you want to predict
        x = self.final_mlp(x)

        return x


model = GNNEncoder()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

labelNames = ["xpos", "ypos"]

train_losses, val_losses = [], []
for epoch in range(epochs):
    with Progress() as p:
        bar = p.add_task(
            "Epoch %d" % (epoch + 1), total=len(train_loader) + len(val_loader)
        )
        # Training
        model.train()
        train_loss: int = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            p.update(bar, advance=1)
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss: int = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
                p.update(bar, advance=1)
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
    print("Traning loss = %f, Validation loss = %f" % (train_loss, val_loss))

# Testing
model.eval()
test_loss: int = 0
true: list = []
preds: list = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        test_loss += loss.item()
        true.append(batch_y.numpy())
        preds.append(predictions.numpy())
test_loss /= len(test_loader)
print("Final test loss: %f" % test_loss)


### --- Plot the loss functions --- ###
plt.figure()
plt.plot(np.linspace(1, epochs, epochs), np.log(train_losses), label="Training loss")
plt.plot(np.linspace(1, epochs, epochs), np.log(val_losses), label="Validation loss")
ax = plt.gca()
# ax.set_ylim(-1.5, 1)
plt.title("Training and validation loss functions")
plt.xlabel("Epoch")
plt.ylabel("Loss (log-scale)")
plt.legend()
plt.savefig("./figures/loss_functions_dropout")


### --- Plot True vs Prediction --- ###
for label in range(2):
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
    plt.savefig("./figures/performance_%s_dropout" % labelNames[label])

