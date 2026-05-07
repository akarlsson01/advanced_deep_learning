from torchvision import transforms
# For DATA SET
import torchvision.datasets as datasets
# For Pytorch methods
import torch
import torch.nn as nn
# For Optimizer
import torch.optim as optim
# FOR DATA LOADER
from torch.utils.data import DataLoader
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import numpy as np
from rich.progress import Progress
# Hyperparameters
LEARNING_RATE = 4e-4
BATCH_SIZE = 128 # Batch size
N_EPOCHS = 100
IMAGE_SIZE = 28
TIME_STEPS = 1000
SAMPLING_TIMESTEPS = 250
# we define a tranform that converts the image to tensor
myTransforms = transforms.Compose([transforms.ToTensor()])
# the MNIST dataset is available through torchvision.datasets
print("loading MNIST digits dataset")
dataset = datasets.MNIST(root="dataset/", transform=myTransforms, download=True)


# let's create a dataloader to load the data in batches
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True) # contains tensors with images and what numbers they are in batches
test_dataset = datasets.MNIST(root='dataset/', train=False, download=False,
transform=myTransforms)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


DIM = 32
DIM_MULTS = (1, 2, 5)
model = Unet(
  dim = DIM,
  dim_mults = DIM_MULTS,
  flash_attn = False,
  channels = 1
)
diffusion = GaussianDiffusion(
  model,
  image_size = IMAGE_SIZE,
  timesteps = TIME_STEPS, # number of steps
  sampling_timesteps = SAMPLING_TIMESTEPS # number of sampling timesteps (using ddim for faster inference [see ddim paper])
)
optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_loss = 0

for epoch in range(N_EPOCHS):
  with Progress() as p:
    bar = p.add_task(f'Epoch {epoch}', total=len(loader.dataset))
    # implement training loop. You get the loss by calling the diffusion function
    # `loss = diffusion(training_images)`
    model.train()

    for images, labels in loader:
      loss = diffusion(images)

      optim.zero_grad()
      loss.backward()
      optim.step()
      p.update(bar, advance=1)

      total_loss += loss.item()

    avg_loss = total_loss / len(loader)

  # you can obtain sampled images (i.e. the backward pass) by calling the sample function
  sampled_images = diffusion.sample(batch_size = 4)
