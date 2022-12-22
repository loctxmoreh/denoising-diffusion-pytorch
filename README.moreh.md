# Running `denoising_diffusion_pytorch` on Moreh AI Framework
![](https://badgen.net/badge/NVIDIA-A100/passed/green)

> This package makes use of `torch.special`, which is only available with `torch>=1.9.1`

## Environment
```bash
conda env create -f a100env.yml
conda activate ddpm
```

## Run
Run 3 example scripts demonstrating how to use this package:

```bash
python3 example.py
python3 example_trainer.py
python3 example_1Dseq.py
```
