import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("model loaded")

text = ['yellow', 'color', 'red']

images = Image.open("yellow.png")
inputs = processor(text=text, images=images, return_tensors="pt", padding=True)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
for i in range(len(text)):
    print(f"text: {text[i]}, probs: {probs[0][i].item()}")
print("done")