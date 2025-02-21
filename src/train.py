import wandb
import torch
import transformers
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE
from model.decoder import Decoder
from utils.loss import calculate_loss
from tqdm import tqdm
from utils.get_datasets import get_datasets

CLIPModel = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


def epoch_loop(batches, name, decoder, device, optimizer=None):
  total_loss = 0
  for batch in tqdm(batches):
    batch_size = batch[0].shape[0]

    caption_encodings = batch[0].to(device)
    image_features = CLIPModel.get_image_features(batch[1].to(device))

    next_word_probs = decoder(caption_encodings, image_features)

    loss = calculate_loss(next_word_probs, caption_encodings)
    average_loss = loss / batch_size
    total_loss += average_loss

    wandb.log(
      {
        f"{name}_batch_size": batch_size,
        f"{name}_loss": average_loss,
      }
    )

    if optimizer:
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  return total_loss / len(batches)


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  train_ds, val_ds, _ = get_datasets()

  print("train_ds len", len(train_ds))
  print("val_ds len", len(val_ds))

  train_batches = torch.utils.data.DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=train_ds.collate,
  )
  val_batches = torch.utils.data.DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=val_ds.collate,
  )

  print(f"Number of training batches: {len(train_batches)}")
  print(f"Number of validation batches: {len(val_batches)}")

  wandb.init(
    project="flickr30k",
    config={
      "batch_size": BATCH_SIZE,
      "learning_rate": LEARNING_RATE,
      "epochs": EPOCHS,
    },
  )

  vocab_size = train_ds.tokenizer.vocab_size
  decoder = Decoder(vocab_size).to(device)

  for epoch in range(EPOCHS):
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

    train_loss = epoch_loop(train_batches, "train", decoder, device, optimizer)
    val_loss = epoch_loop(val_batches, "val", decoder, device)

    wandb.log(
      {
        "epoch": epoch,
        "mean_train_loss": train_loss,
        "mean_val_loss": val_loss,
      }
    )

  # Save model weights
  torch.save(decoder.state_dict(), "model_weights.pt")

  # Save model weights to wandb
  artifact = wandb.Artifact("model_weights", type="model")
  artifact.add_file("model_weights.pt")
  wandb.log_artifact(artifact)

  # Finish the wandb run
  wandb.finish()
