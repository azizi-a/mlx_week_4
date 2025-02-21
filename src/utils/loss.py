import torch.nn as nn

criterion = nn.CrossEntropyLoss()


def calculate_loss(predictions, targets):
  # print('predictions shape', predictions.shape)
  # print('targets shape', targets.shape)
  predictions = predictions[:, 1:, :].reshape(-1, predictions.size(-1))
  targets = targets.reshape(-1)
  #   print("reshaped predictions shape", predictions.shape)
  #   print("reshaped targets shape", targets.shape)
  loss = criterion(predictions, targets)
  return loss
