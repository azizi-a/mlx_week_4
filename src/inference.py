import torch
import transformers
import torchvision
from utils.get_datasets import get_datasets
from model.decoder import Decoder

CLIPModel = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

_, _, test_ds = get_datasets()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decoder = Decoder(test_ds.tokenizer.vocab_size).to(device)
decoder.load_state_dict(torch.load("model_weights.pt", map_location=device))
decoder.eval()


def get_test_item():
  random_idx = torch.randint(0, len(test_ds), (1,)).item()
  random_item = test_ds[random_idx]

  # Get image and caption
  image = random_item[1]
  caption = random_item[0]

  return image, caption


def predict_caption(image):
  with torch.no_grad():
    image_features = CLIPModel.get_image_features(image.unsqueeze(0))
  start_token = test_ds.tokenizer.bos_token_id
  end_token = test_ds.tokenizer.eos_token_id

  text_input = torch.tensor([start_token]).unsqueeze(0)

  # Get predictions
  with torch.no_grad():
    predicted_word_tokens = text_input
    while predicted_word_tokens[-1][-1].item() != end_token and predicted_word_tokens.shape[1] < 32:
      next_word_probs = decoder(predicted_word_tokens, image_features)
      next_word_tokens = next_word_probs.argmax(dim=2)
      predicted_word_tokens = torch.cat((predicted_word_tokens, next_word_tokens), dim=1)

  predicted_words = test_ds.tokenizer.decode(predicted_word_tokens[0])
  return predicted_words


def display_test_image(image, title):
  # Denormalize the image tensor
  denorm = torchvision.transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
  )
  image_denorm = denorm(image)

  # Convert to PIL Image
  image_pil = torchvision.transforms.ToPILImage()(image_denorm)

  # Display the image
  import matplotlib.pyplot as plt

  plt.imshow(image_pil)
  plt.title(title)
  plt.axis("off")
  plt.show()


if __name__ == "__main__":
  image, caption = get_test_item()
  predicted_caption = predict_caption(image)

  caption = (
    test_ds.tokenizer.decode(caption).replace(test_ds.tokenizer.bos_token, "").replace(test_ds.tokenizer.eos_token, "")
  )
  predicted_caption = predicted_caption.replace(test_ds.tokenizer.bos_token, "").replace(
    test_ds.tokenizer.eos_token, ""
  )

  print("Original caption:", caption)
  print("Predicted words:", predicted_caption)

  display_test_image(image, predicted_caption)
