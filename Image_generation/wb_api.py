from gradio_client import Client

client = Client("black-forest-labs/FLUX.1-schnell")
result = client.predict(
		prompt="A big camel walking on a beach",
		seed=0,
		randomize_seed=True,
		width=1024,
		height=1024,
		num_inference_steps=4,
		api_name="/infer"
)
print(result)