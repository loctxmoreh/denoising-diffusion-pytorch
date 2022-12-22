import sys
import os

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


image_path = sys.argv[1]
assert os.path.exists(image_path), f"Image dir {image_path} does not exist"

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1'            # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    image_path,
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 1000,           # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True                        # turn on mixed precision
)

trainer.train()
