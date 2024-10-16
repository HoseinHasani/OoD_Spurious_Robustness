import torch
from diffusers import FluxPipeline
import os 
import numpy as np
import time
nn = 100
nnn = 1

min_step = 4
max_step = 8

#pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)

pipe = pipe.to("cuda")
    
    
for j in range(40, nn):

    prompt = "Picture of a desert"
    pic_path = f"data/{prompt}"
    
    os.makedirs(pic_path, exist_ok=True)
    

    for i in range(nnn):
        
        t0 = time.time()
        seed = int(t0) % 1000000
        np.random.seed(seed)
    
        num_inference_steps = np.random.randint(min_step, max_step)
        guidance_scale = 3 + 5 * np.random.rand()
    
        image = pipe(prompt,
                    height=224,
                    width=224,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    max_sequence_length=256,
                    generator=torch.Generator("cuda").manual_seed(seed + 5)
                     ).images[0]
        
        image.save(f"{pic_path}/{j*nn + i}_{seed}.png")
        
        print(i, int(time.time() - t0), seed, num_inference_steps)

    prompt = "Picture of a grassland"
    pic_path = f"data/{prompt}"
    
    os.makedirs(pic_path, exist_ok=True)
    
    for i in range(nnn):
        
        t0 = time.time()
        seed = int(t0) % 1000000
        np.random.seed(seed)
    
        num_inference_steps = np.random.randint(min_step, max_step)
        guidance_scale = 3 + 5 * np.random.rand()
    
        image = pipe(prompt,
                    height=224,
                    width=224,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    max_sequence_length=256,
                    generator=torch.Generator("cuda").manual_seed(seed + 5)
                     ).images[0]
        
        image.save(f"{pic_path}/{j*nn + i}_{seed}.png")
        
        print(i, int(time.time() - t0), seed, num_inference_steps)
        
        
    prompt = "A cow in a grassland"
    pic_path = f"data/{prompt}"
    
    os.makedirs(pic_path, exist_ok=True)
    
    for i in range(4*nnn):
        
        t0 = time.time()
        seed = int(t0) % 1000000
        np.random.seed(seed)
    
        num_inference_steps = np.random.randint(min_step, max_step)
        guidance_scale = 3 + 5 * np.random.rand()
    
        image = pipe(prompt,
                    height=224,
                    width=224,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    max_sequence_length=256,
                    generator=torch.Generator("cuda").manual_seed(seed + 5)
                     ).images[0]
        
        image.save(f"{pic_path}/{j*nn + i}_{seed}.png")
        
        print(i, int(time.time() - t0), seed, num_inference_steps)
        
        
    prompt = "A horse in a grassland"
    pic_path = f"data/{prompt}"
    
    os.makedirs(pic_path, exist_ok=True)
    
    for i in range(2*nnn):
        
        t0 = time.time()
        seed = int(t0) % 1000000
        np.random.seed(seed)
    
        num_inference_steps = np.random.randint(min_step, max_step)
        guidance_scale = 3 + 5 * np.random.rand()
    
        image = pipe(prompt,
                    height=224,
                    width=224,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    max_sequence_length=256,
                    generator=torch.Generator("cuda").manual_seed(seed + 5)
                     ).images[0]
        
        image.save(f"{pic_path}/{j*nn + i}_{seed}.png")
        
        print(i, int(time.time() - t0), seed, num_inference_steps)
        
    prompt = "A camel in a green grassland"
    pic_path = f"data/{prompt}"
    
    os.makedirs(pic_path, exist_ok=True)
    
    for i in range(nnn):
        
        t0 = time.time()
        seed = int(t0) % 1000000
        np.random.seed(seed)
    
        num_inference_steps = np.random.randint(min_step, max_step)
        guidance_scale = 3 + 5 * np.random.rand()
    
        image = pipe(prompt,
                    height=224,
                    width=224,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    max_sequence_length=256,
                    generator=torch.Generator("cuda").manual_seed(seed + 5)
                     ).images[0]
        
        image.save(f"{pic_path}/{j*nn + i}_{seed}.png")
        
        print(i, int(time.time() - t0), seed, num_inference_steps)
        
        
    prompt = "A cow in a desert"
    pic_path = f"data/{prompt}"
    
    os.makedirs(pic_path, exist_ok=True)
    
    for i in range(nnn):
        
        t0 = time.time()
        seed = int(t0) % 1000000
        np.random.seed(seed)
    
        num_inference_steps = np.random.randint(min_step, max_step)
        guidance_scale = 3 + 5 * np.random.rand()
    
        image = pipe(prompt,
                    height=224,
                    width=224,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    max_sequence_length=256,
                    generator=torch.Generator("cuda").manual_seed(seed + 5)
                     ).images[0]
        
        image.save(f"{pic_path}/{j*nn + i}_{seed}.png")
        
        print(i, int(time.time() - t0), seed, num_inference_steps)
        
        
    prompt = "A horse in a desert"
    pic_path = f"data/{prompt}"
    
    os.makedirs(pic_path, exist_ok=True)
    
    for i in range(2*nnn):
        
        t0 = time.time()
        seed = int(t0) % 1000000
        np.random.seed(seed)
    
        num_inference_steps = np.random.randint(min_step, max_step)
        guidance_scale = 3 + 5 * np.random.rand()
    
        image = pipe(prompt,
                    height=224,
                    width=224,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    max_sequence_length=256,
                    generator=torch.Generator("cuda").manual_seed(seed + 5)
                     ).images[0]
        
        image.save(f"{pic_path}/{j*nn + i}_{seed}.png")
        
        print(i, int(time.time() - t0), seed, num_inference_steps)
        
    prompt = "A camel in a desert"
    pic_path = f"data/{prompt}"
    
    os.makedirs(pic_path, exist_ok=True)
    
    for i in range(4*nnn):
        
        t0 = time.time()
        seed = int(t0) % 1000000
        np.random.seed(seed)
    
        num_inference_steps = np.random.randint(min_step, max_step)
        guidance_scale = 3 + 5 * np.random.rand()
    
        image = pipe(prompt,
                    height=224,
                    width=224,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    max_sequence_length=256,
                    generator=torch.Generator("cuda").manual_seed(seed + 5)
                     ).images[0]
        
        image.save(f"{pic_path}/{j*nn + i}_{seed}.png")
        
        print(i, int(time.time() - t0), seed, num_inference_steps)