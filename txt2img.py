# make sure you're logged in with `huggingface-cli login`
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-5")
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"

# image = pipe(prompt).sample[0]        # diffusers==0.12.1, 'StableDiffusionPipelineOutput' object has no attribute 'sample'
result = pipe(prompt)
image = result.images[0]                # len(images): 1
nsfw  = result.nsfw_content_detected[0] # true or false

image.save("runs/astronaut_rides_horse.png")

print(f"==> nsfw_content_detected: {nsfw}")
print(f"==> image saved.")