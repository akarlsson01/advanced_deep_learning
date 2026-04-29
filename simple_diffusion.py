import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import seaborn as sns # a useful plotting library on top of matplotlib
from tqdm.auto import tqdm # a nice progress bar
# # This is a simple example of a diffusion model in 1D.
# generate a dataset of 1D data from a mixture of two Gaussians
# this is a simple example, but you can use any distribution
data_distribution = torch.distributions.mixture_same_family.MixtureSameFamily(
torch.distributions.Categorical(torch.tensor([1, 2])),
torch.distributions.Normal(torch.tensor([-4., 4.]), torch.tensor([1., 1.]))
)
dataset = data_distribution.sample(torch.Size([10000])) # create training data set
dataset_validation = data_distribution.sample(torch.Size([1000])) # create validation data set
fig, ax = plt.subplots(1, 1)
sns.histplot(dataset)
plt.show()
print(dataset.shape)
# we will keep these parameters fixed throughout
# these parameters should give you an acceptable result
# but feel free to play with them
TIME_STEPS = 250
BETA = torch.tensor(0.02)
N_EPOCHS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 0.8e-4
# define the neural network that predicts the amount of noise that was
# added to the data
# the network should have two inputs (the current data and the time step)
# and one output (the predicted noise)
g = nn.Sequential(
  nn.Linear(2, 128),
  nn.ReLU(),
  nn.Linear(128, 128),
  nn.ReLU(),
  nn.Linear(128, 1)
)
epochs = tqdm(range(N_EPOCHS)) # this makes a nice progress bar
optimizer = torch.optim.Adam(g.parameters(), lr=LEARNING_RATE)
beta = torch.linspace(1e-4, 0.02, TIME_STEPS)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)    
for e in epochs: # loop over epochs
  g.train()
  # loop through batches of the dataset, reshuffling it each epoch
  indices = torch.randperm(dataset.shape[0])
  shuffled_dataset = dataset[indices]
  for i in range(0, shuffled_dataset.shape[0] - BATCH_SIZE, BATCH_SIZE):
    x0 = shuffled_dataset[i:i + BATCH_SIZE]

    t = torch.randint(0, TIME_STEPS, (BATCH_SIZE,))
    noise = torch.randn_like(x0)

    a_bar = alpha_bar[t]
    xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise

    inp = torch.cat([xt.unsqueeze(1), t.unsqueeze(1).float()], dim=1)
    pred = g(inp).squeeze()

    loss = nn.MSELoss()(pred, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def sample_reverse(g, count):
  """
  Sample from the model by applying the reverse diffusion process
  Here, implement algorithm 2 of the DDPM paper
  (https://arxiv.org/abs/2006.11239)
  Parameters
  ----------
  g : torch.nn.Module
  The neural network that predicts the noise added to the data
  count : int
  The number of samples to generate in parallel
  Returns
  -------
  x : torch.Tensor
  The final sample from the model
  """
  g.eval()

  x = torch.randn(count)

  for t in reversed(range(TIME_STEPS)):
      t_tensor = torch.full((count,), t, dtype=torch.float32)

      inp = torch.stack([x, t_tensor], dim=1)
      pred_noise = g(inp).squeeze()

      a = alpha[t]
      a_bar = alpha_bar[t]
      b = beta[t]

      if t > 0:
          z = torch.randn_like(x)
      else:
          z = 0

      x = (1 / torch.sqrt(a)) * (
          x - (1 - a) / torch.sqrt(1 - a_bar) * pred_noise
      ) + torch.sqrt(b) * z

  return x
    


samples = sample_reverse(g, 1000)
samples = samples.detach().numpy()
# plot the samples
fig, ax = plt.subplots(1, 1)
bins = np.linspace(-10, 10, 50)
sns.kdeplot(dataset, ax=ax, color='blue', label='True distribution', linewidth=2)
sns.histplot(samples, ax=ax, bins=bins, color='red', label='Sampled distribution',
stat='density')
ax.legend()
ax.set_xlabel('Sample value')
ax.set_ylabel('Sample count')
