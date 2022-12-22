import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 32
).to(device)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 128,
    timesteps = 1000,
    objective = 'pred_v'
).to(device)

training_seq = torch.rand(8, 32, 128).to(device) # features are normalized from 0 to 1
loss = diffusion(training_seq)
loss.backward()

# after a lot of training

sampled_seq = diffusion.sample(batch_size = 4)
#print(sampled_seq.shape) # (4, 32, 128)

sampled_shape = sampled_seq.detach().shape
assert sampled_shape == (4, 32, 128)
