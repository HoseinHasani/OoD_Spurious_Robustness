from gradio_client import Client
import numpy as np
import time

t_sleep = 90
n_images = 100

prompt_list = [
              "A cat beside a woman",
              "A woman hugging a cat",
              "A woman is playing with a cat",
              ]


for i in range(n_images):
    
    t0 = time.time()
    seed = int(t0) % 1000000
    np.random.seed(seed)
    steps = np.random.randint(3, 6)
    
    prompt_ind = np.random.choice(len(prompt_list), 1).item()
    prompt = prompt_list[prompt_ind]
    
    client = Client("black-forest-labs/FLUX.1-schnell", download_files="pics3")
    result = client.predict(
    		prompt=prompt,
    		seed=seed,
    		randomize_seed=True,
    		width=1024,
    		height=1024,
    		num_inference_steps=steps,
    		api_name="/infer"
    )
    #print(result)
    time.sleep(t_sleep + 4 * np.random.rand())
    
    print(i, int(time.time() - t0), seed, steps, prompt)
    
    
