import torch
from diffusers import FluxPipeline
from huggingface_hub import login

login(token="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")


pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

prompt = "A big camel walking on a beach"
image = pipe(
    prompt,
    height=512,
    width=512,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("camel_beach.png")