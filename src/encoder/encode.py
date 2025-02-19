
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

def tokenize_text(caption_batch):
  caption_count = len(caption_batch[0])

  # Flatten the captions into a single list
  flattened_captions = [caption for captions in caption_batch for caption in captions]

  # Process all captions through CLIP processor
  caption_inputs = CLIPProcessor.tokenizer(flattened_captions, padding=True, truncation=True, return_tensors="pt")

  # Get text embeddings from CLIP model
  with torch.no_grad():
    caption_features = CLIPModel.get_text_features(**caption_inputs)

  return caption_inputs.input_ids, caption_features, caption_count


def encode_data_batch(data_batch):
  image_features = encode_image(data_batch['image'])
  caption_encodings, caption_features, caption_count = tokenize_text(data_batch['caption'])
  repeated_image_features = image_features.repeat_interleave(5, dim=0)

  encoded_data_batch = {
    'image_features': repeated_image_features,
    'caption_encodings': caption_encodings,
    'caption_features': caption_features,
    'caption_count': caption_count
  }

  return encoded_data_batch

def get_vocab_size():
  return CLIPProcessor.tokenizer.vocab_size

