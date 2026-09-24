#!/usr/bin/env python3
"""
Generate image captions using a pre-trained BLIP Vision-Language Model.
"""
import transformers
import PIL


def image_captioner(model, image_path, max_new_tokens):
    """
    Generates a textual description (caption) of a given image using a
    pre-trained BLIP Vision-Language Model.

    Args:
        model (str): Name of the pre-trained image captioning model to
            use.
        image_path (str): Path to the image file to caption.
        max_new_tokens (int): Maximum number of tokens to generate.

    Returns:
        caption (str): Generated textual description of the image.
    """
    processor = transformers.BlipProcessor.from_pretrained(model)
    blip_model = transformers.BlipForConditionalGeneration.from_pretrained(
        model
    )

    image = PIL.Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    generated_ids = blip_model.generate(
        **inputs, max_new_tokens=max_new_tokens
    )

    caption = processor.decode(generated_ids[0], skip_special_tokens=True)

    return caption
