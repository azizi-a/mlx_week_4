from data.flick_dataset import Flick
import torch
import os


def get_datasets():
  if os.path.exists("data_cache/train_ds.pt"):
    train_ds = torch.load("data_cache/train_ds.pt", weights_only=False)
  else:
    train_ds = Flick(split="train")
    os.makedirs("data_cache", exist_ok=True)
    torch.save(train_ds, "data_cache/train_ds.pt")

  if os.path.exists("data_cache/val_ds.pt"):
    val_ds = torch.load("data_cache/val_ds.pt", weights_only=False)
  else:
    val_ds = Flick(split="val")
    os.makedirs("data_cache", exist_ok=True)
    torch.save(val_ds, "data_cache/val_ds.pt")

  if os.path.exists("data_cache/test_ds.pt"):
    test_ds = torch.load("data_cache/test_ds.pt", weights_only=False)
  else:
    test_ds = Flick(split="test")
    os.makedirs("data_cache", exist_ok=True)
    torch.save(test_ds, "data_cache/test_ds.pt")

  return train_ds, val_ds, test_ds
