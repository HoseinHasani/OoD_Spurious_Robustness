
from gradio_client import Client
import numpy as np
import time

t_sleep = 90
n_images = 100

prompt_list = [
              "A cat beside a man",
              "A man hugging a cat",
              "A man is playing with a cat",
              ]


for i in range(n_images):
    
    t0 = time.time()
    seed = int(t0) % 1000000
    np.random.seed(seed)
    steps = np.random.randint(3, 6)
    
    prompt_ind = np.random.choice(len(prompt_list), 1).item()
    prompt = prompt_list[prompt_ind]
    
    client = Client("stabilityai/stable-diffusion-3-medium", download_files="pics4")
    result = client.predict(
    		prompt=prompt,
            negative_prompt='low quality image',
    		seed=seed,
    		randomize_seed=True,
    		width=1024,
    		height=1024,
    		guidance_scale=5,
    		num_inference_steps=28,
    		api_name="/infer"
    )
    #print(result)
    time.sleep(t_sleep + 4 * np.random.rand())
    
    print(i, int(time.time() - t0), seed, steps, prompt)
    
    
