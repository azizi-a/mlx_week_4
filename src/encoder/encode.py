
import transformers
import torch

CLIPModel = transformers.CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
CLIPProcessor = transformers.CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

def encode_image(image_batch):
  pixel_vals = CLIPProcessor.image_processor(image_batch)
  pixel_vals_tensor = torch.tensor(pixel_vals.pixel_values)

  with torch.no_grad():
      image_features = CLIPModel.get_image_features(pixel_vals_tensor)
  
  return image_features

def encode_text(caption_batch):
  batch_size = len(caption_batch)
  caption_count = len(caption_batch[0])

  # Flatten the captions into a single list
  flattened_captions = [caption for captions in caption_batch for caption in captions]

  # Process all captions through CLIP processor
  text_inputs = CLIPProcessor.tokenizer(flattened_captions, padding=True, truncation=True, return_tensors="pt")

  # Get text embeddings from CLIP model
  with torch.no_grad():
      text_features = CLIPModel.get_text_features(**text_inputs)

  return text_features, caption_count


def encode_data_batch(data_batch):
  image_features = encode_image(data_batch['image'])
  text_features, caption_count = encode_text(data_batch['caption'])
  repeated_image_features = image_features.repeat_interleave(5, dim=0)

  encoded_data_batch = {
    'image': repeated_image_features,
    'text': text_features,
    'caption_count': caption_count
  }

  return encoded_data_batch

