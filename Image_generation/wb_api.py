from gradio_client import Client
import numpy as np


seed = np.random.randint(0, 10)
steps = np.random.randint(2, 6)
print(seed, steps)
prompt = "A german shepherd beside a woman"
prompt = "A Doberman beside a man"

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
print(result)