from gradio_client import Client
import numpy as np
import time

t_sleep = 6
n_images = 100

prompt_list = [
              "A german shepherd beside a woman",
              "A woman hugging a german shephered dog",
              "A woman is playing with a german shephered",
              ]


for i in range(n_images):
    
    t0 = time.time()
    seed = int(t0) % 1000000
    np.random.seed(seed)
    steps = np.random.randint(2, 6)
    
    prompt_ind = np.random.choice(len(prompt_list), 1).item()
    prompt = prompt_list[prompt_ind]
    
    client = Client("black-forest-labs/FLUX.1-schnell", download_files="pics")
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
    time.sleep(t_sleep + 2 * np.random.rand())
    
    print(i, int(time.time() - t0), seed, steps, prompt)
    
    
