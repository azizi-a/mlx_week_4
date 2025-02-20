import wandb
import torch
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE
from data.data_downloader import DataDownloader
from data.data_batcher import create_batches_for_splits
from encoder.encode import encode_data_batch, get_vocab_size
from model.decoder import Decoder
from utils.loss import calculate_loss
from tqdm import tqdm


def epoch_loop(batches, name, decoder, optimizer=None):
    total_loss = 0
    for batch in tqdm(batches):
        batch_size = batch["image"].shape[0]

        encoded_batch = encode_data_batch(batch)

        next_word_probs = decoder.forward(
            encoded_batch["caption_encodings"],
            encoded_batch["image_features"],
        )

        loss = calculate_loss(
            next_word_probs,
            encoded_batch["caption_encodings"],
        )
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
    ds = DataDownloader()
    train_batches, val_batches, test_batches = create_batches_for_splits(ds)

    print(f"Number of training batches: {len(train_batches)}")
    print(f"Number of validation batches: {len(val_batches)}")
    print(f"Number of test batches: {len(test_batches)}")

    wandb.init(
        project="flickr30k",
        config={
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
        },
    )

    vocab_size = get_vocab_size()
    decoder = Decoder(vocab_size)

    for epoch in range(EPOCHS):
        optimizer = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

        train_loss = epoch_loop(train_batches, "train", decoder, optimizer)
        val_loss = epoch_loop(val_batches, "val", decoder)

        wandb.log(
            {
                "epoch": epoch,
                "mean_train_loss": train_loss,
                "mean_val_loss": val_loss,
            }
        )

    # Save model weights
    torch.save(decoder.state_dict(), "model_weights.pt")

    # Finish the wandb run
    wandb.finish()
