import datasets
import torchvision
from PIL import Image
import numpy as np

def resize_with_padding(image, target_size=224):
    # Convert to PIL Image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.uint8(image))
    
    # Get original dimensions
    original_width, original_height = image.size
    
    # Calculate scaling factor to maintain aspect ratio
    ratio = min(target_size / original_width, target_size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    
    # Create transform pipeline
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((new_height, new_width)),
        torchvision.transforms.Pad(padding=((target_size - new_width)//2, 
                      (target_size - new_height)//2),
              fill=0),  # Black padding
        torchvision.transforms.CenterCrop(target_size)
    ])
    
    # Transform the image
    return transform(image)

# Apply the resizing to the dataset
def process_example(example):
    example["image"] = resize_with_padding(example["image"])
    return example

def process_dataset():
    try:
        ds = datasets.load_from_disk("data_cache/processed_dataset")
    except:
        ds = datasets.load_dataset("nlphuji/flickr30k", cache_dir="./data_cache")
        print("Processing dataset...")
        ds = ds.map(process_example)
        ds.save_to_disk("data_cache/processed_dataset")
    return ds

if __name__ == "__main__":
    try:
        ds = process_dataset()
    except Exception as e:
        print(f"Error processing dataset: {e}")
        raise e
