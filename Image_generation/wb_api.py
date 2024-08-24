from gradio_client import Client

client = Client("black-forest-labs/FLUX.1-schnell", download_files="pics")
result = client.predict(
		prompt="A Cow walking in grass field",
		seed=0,
		randomize_seed=True,
		width=1024,
		height=1024,
		num_inference_steps=4,
		api_name="/infer"
)
print(result)