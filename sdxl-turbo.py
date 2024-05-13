import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
)

prompt = "2 best friends in a garden full of roses"
pipe.enable_attention_slicing()
image = pipe(prompt).images[0]
image
