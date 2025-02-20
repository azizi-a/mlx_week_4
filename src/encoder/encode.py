import transformers
import torch
import numpy as np

MODEL_NAME = "openai/clip-vit-base-patch32"

CLIPModel = transformers.CLIPModel.from_pretrained(MODEL_NAME)
CLIPProcessor = transformers.CLIPProcessor.from_pretrained(MODEL_NAME)


def encode_image(image_batch):
    pixel_vals = CLIPProcessor.image_processor(image_batch)
    pixel_vals_tensor = torch.tensor(pixel_vals.pixel_values)

    with torch.no_grad():
        image_features = CLIPModel.get_image_features(pixel_vals_tensor)

    return image_features


def tokenize_text(caption_batch):
    ca = np.array(caption_batch)
    caption_count = ca.shape[0]

    # Flatten the captions into a single list
    flattened_captions = []
    for captions in caption_batch:
        flattened_captions.extend(captions)

    # Process all captions through CLIP processor
    caption_inputs = CLIPProcessor.tokenizer(
        flattened_captions, padding=True, truncation=True, return_tensors="pt"
    )

    # Get text embeddings from CLIP model
    with torch.no_grad():
        caption_features = CLIPModel.get_text_features(**caption_inputs)

    return caption_inputs.input_ids, caption_features, caption_count


def encode_data_batch(data_batch):
    image_batch = data_batch["image"]
    caption_batch = data_batch["caption"]

    image_features = image_batch
    text_outputs = tokenize_text(caption_batch)
    caption_encodings, caption_features, caption_count = text_outputs
    repeated_image_features = image_features.repeat_interleave(caption_count, dim=0)

    encoded_data_batch = {
        "image_features": repeated_image_features,
        "caption_encodings": caption_encodings,
        "caption_features": caption_features,
        "caption_count": caption_count,
    }

    return encoded_data_batch


def get_vocab_size():
    return CLIPProcessor.tokenizer.vocab_size
