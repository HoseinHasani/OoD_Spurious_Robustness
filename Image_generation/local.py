import torch
from diffusers import FluxPipeline
import os 
import numpy as np
import time

prompt = "A cow in a grass field"
pic_path = f"data/{prompt}"

os.makedirs(pic_path, exist_ok=True)

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

for i in range(10):
    
    t0 = time.time()
    seed = int(t0) % 1000000
    np.random.seed(seed)

    num_inference_steps = np.random.randint(30, 70)
    guidance_scale = 3 + 5 * np.random.rand()

    image = pipe(prompt,
                height=224,
                width=224,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_sequence_length=256,
                generator=torch.Generator("cuda").manual_seed(seed + 5)
                 ).images[0]
    
    image.save(f"{pic_path}/{i}_{seed}.png")
    
    print(i, int(time.time() - t0), seed, num_inference_steps)