import torch
from diffusers import FluxPipeline
import os 
import numpy as np

prompt = "A cow in a grass field"

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

for i in range(100):

    n_steps = np.random.randint(25, 55)

    image = pipe(prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=n_steps,
                max_sequence_length=512,
                 ).images[0]
    image.save("flux.png")