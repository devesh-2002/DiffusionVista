from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
import numpy as np
import torch
import cv2
from PIL import Image

prompt = "2 best friends in a garden full of roses"
negative_prompt = "low quality, bad quality"


original_image = load_image(
    "image.png"
)

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16, use_safetensors=True
)
pipe.enable_model_cpu_offload()

image = np.array(original_image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

image = pipe(
    prompt, negative_prompt=negative_prompt, controlnet_conditioning_scale=0.5, image=canny_image, guess_mode=True,width=2048,height=2048
).images[0]
image
make_image_grid([original_image, canny_image, image], rows=1, cols=3)
