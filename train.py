import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from musicgenerator import MelodyGenerator
from melodypreprocessor import MelodyPreprocessor
from transformer import Transformer

# Global parameters
EPOCHS = 10
BATCH_SIZE = 32
DATA_PATH = "dataset.json"
MAX_POSITIONS_IN_POSITIONAL_ENCODING = 100

# Loss function and optimizer
loss_function = nn.CrossEntropyLoss(reduction="none")


def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=0.001)


def train(train_dataset, transformer, epochs, device):
    """
    Trains the Transformer model on a given dataset for a specified number of epochs.

    Parameters:
        train_dataset (DataLoader): The training dataset.
        transformer (Transformer): The Transformer model instance.
        epochs (int): The number of epochs to train the model.
        device (torch.device): The device to use (CPU or GPU).
    """
    optimizer = get_optimizer(transformer)
    print("Training the model...")

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (input_batch, target_batch) in enumerate(train_dataset):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            batch_loss = _train_step(
                input_batch, target_batch, transformer, optimizer, device)
            total_loss += batch_loss.item()

            print(
                f"Epoch {epoch + 1} Batch {batch_idx + 1} Loss {batch_loss.item():.4f}")


def _train_step(input_batch, target_batch, transformer, optimizer, device):
    """
    Performs a single training step for the Transformer model.

    Parameters:
        input_batch (torch.Tensor): The input sequences.
        target_batch (torch.Tensor): The target sequences.
        transformer (Transformer): The Transformer model instance.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        device (torch.device): The device to use (CPU or GPU).

    Returns:
        torch.Tensor: The loss value for the training step.
    """
    transformer.train()

    # Prepare target input and target real
    target_input = _right_pad_sequence_once(target_batch[:, :-1], device)
    target_real = _right_pad_sequence_once(target_batch[:, 1:], device)

    # Forward pass
    # change is needed
    predictions = transformer(input_batch, target_input, None, None, None)

    # Compute loss
    loss = _calculate_loss(target_real, predictions)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def _calculate_loss(real, pred):
    """
    Computes the loss between the real and predicted sequences.

    Parameters:
        real (torch.Tensor): The actual target sequences.
        pred (torch.Tensor): The predicted sequences by the model.

    Returns:
        torch.Tensor: The average loss value.
    """
    # Reshape predictions and real for loss computation
    pred = pred.view(-1, pred.size(-1))
    real = real.view(-1)

    loss = loss_function(pred, real)

    # Create mask for padded values (real == 0)
    mask = real != 0
    loss = loss * mask.float()

    # Calculate average loss excluding padded positions
    total_loss = torch.sum(loss)
    num_non_padded = torch.sum(mask)

    return total_loss / num_non_padded


def _right_pad_sequence_once(sequence, device):
    """
    Pads a sequence with a single zero at the end.

    Parameters:
        sequence (torch.Tensor): The sequence to be padded.
        device (torch.device): The device to use (CPU or GPU).

    Returns:
        torch.Tensor: The padded sequence.
    """
    padding = torch.zeros((sequence.size(0), 1),
                          dtype=sequence.dtype, device=device)
    return torch.cat((sequence, padding), dim=1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess dataset
    melody_preprocessor = MelodyPreprocessor(DATA_PATH, batch_size=BATCH_SIZE)
    # train_dataset = DataLoader(melody_preprocessor.create_training_dataset(
    # ), batch_size=BATCH_SIZE, shuffle=True)
    train_dataset = melody_preprocessor.create_training_dataset(
    )
    vocab_size = melody_preprocessor.number_of_tokens_with_padding

    # Initialize Transformer model
    transformer_model = Transformer(
        num_layers=2,
        d_model=64,
        num_heads=2,
        d_feedforward=128,
        input_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        max_num_positions_in_pe_encoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
        max_num_positions_in_pe_decoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
        dropout_rate=0.1,
    ).to(device)

    # Train the model
    train(train_dataset, transformer_model, EPOCHS, device)

    # Generate a melody
    print("Generating a melody...")
    melody_generator = MelodyGenerator(
        transformer_model, melody_preprocessor.tokenizer)
    start_sequence = ["C4-1.0", "D4-1.0", "E4-1.0", "C4-1.0"]
    new_melody = melody_generator.generate(start_sequence)
    print(f"Generated melody: {new_melody}")
