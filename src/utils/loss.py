import torch.nn as nn

criterion = nn.CrossEntropyLoss()


def calculate_loss(predictions, targets):
  predictions = predictions[:, 1:, :].reshape(-1, predictions.size(-1))
  targets = targets.reshape(-1)
  loss = criterion(predictions, targets)
  return loss
