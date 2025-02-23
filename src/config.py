# Training hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 10

# Model hyperparameters
IMAGE_SIZE = 224
PATCH_SIZE = 14
NUM_PATCHES = IMAGE_SIZE**2 // PATCH_SIZE**2
EMBEDDING_DIM = 512
LATENT_DIM = 128
DECODER_ATTENTION_BLOCK_COUNT = 10
