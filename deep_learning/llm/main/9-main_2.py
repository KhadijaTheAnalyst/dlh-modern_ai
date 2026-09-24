#!/usr/bin/env python3

from PIL import Image
import matplotlib.pyplot as plt
image_captioner = __import__('9-image_captioner').image_captioner

image_path = "animal.PNG"

caption_model = "Salesforce/blip-image-captioning-base"

image = Image.open(image_path)
plt.imshow(image)
plt.axis('off')
plt.show()

caption_result = image_captioner(caption_model, image_path, 10)
print("Caption result:", caption_result)
