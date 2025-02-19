import torch
from config import BATCH_SIZE

def create_batches(dataset):
    # Create DataLoader with specified batch size
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    return dataloader

def create_batches_for_each_split(dataset):
    # Create batches for each split type and cache them
    try:
        train_batches = torch.load('data_cache/train_batches.pt')
        validation_batches = torch.load('data_cache/validation_batches.pt') 
        test_batches = torch.load('data_cache/test_batches.pt')
    except:
        train_batches = create_batches(dataset.filter(lambda x: x['split'] == 'train'))
        validation_batches = create_batches(dataset.filter(lambda x: x['split'] == 'val'))
        test_batches = create_batches(dataset.filter(lambda x: x['split'] == 'test'))
        
        # Cache the batches
        torch.save(train_batches, 'data_cache/train_batches.pt')
        torch.save(validation_batches, 'data_cache/validation_batches.pt')
        torch.save(test_batches, 'data_cache/test_batches.pt')
        
    return train_batches, validation_batches, test_batches