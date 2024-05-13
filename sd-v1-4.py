from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionPipeline
import torch


pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
)
pipeline.to("cuda")

model_id = "stabilityai/sd-x2-latent-upscaler"
upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
upscaler.to("cuda")

prompt = "2 best friends in a garden full of roses"
generator = torch.manual_seed(33)

low_res_latents = pipeline(prompt, generator=generator, output_type="latent").images

with torch.no_grad():
    image = pipeline.decode_latents(low_res_latents)
image = pipeline.numpy_to_pil(image)[0]

image.save("./a1.png")

upscaled_image = upscaler(
    prompt=prompt,
    image=low_res_latents,
    num_inference_steps=20,
    guidance_scale=0,
    generator=generator,
).images[0]
upscaled_image
upscaled_image.save("./a2.png")
