from collections import Counter
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
import torch


class MelodyPreprocessor:
    """
    A class for preprocessing melodies for a Transformer model.

    This class takes melodies, tokenizes and encodes them, and prepares
    PyTorch datasets for training sequence-to-sequence models.
    """

    def __init__(self, dataset_path, batch_size=32):
        """
        Initializes the MelodyPreprocessor.

        Parameters:
            dataset_path (str): Path to the dataset file.
            batch_size (int): Size of each batch in the dataset.
        """
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.tokenizer = None
        self.max_melody_length = None
        self.number_of_tokens = None

    @property
    def number_of_tokens_with_padding(self):
        """
        Returns the number of tokens in the vocabulary including padding.

        Returns:
            int: The number of tokens in the vocabulary including padding.
        """
        return self.number_of_tokens + 1

    def create_training_dataset(self):
        """
        Preprocesses the melody dataset and creates sequence-to-sequence
        training data.

        Returns:
            DataLoader: A PyTorch DataLoader containing input-target pairs
            suitable for training a sequence-to-sequence model.
        """
        dataset = self._load_dataset()
        parsed_melodies = [self._parse_melody(melody) for melody in dataset]
        tokenized_melodies = self._tokenize_and_encode_melodies(
            parsed_melodies)
        self._set_max_melody_length(tokenized_melodies)
        self._set_number_of_tokens()
        input_sequences, target_sequences = self._create_sequence_pairs(
            tokenized_melodies)
        return self._convert_to_dataloader(input_sequences, target_sequences)

    def _load_dataset(self):
        """
        Loads the melody dataset from a JSON file.

        Returns:
            list: A list of melodies from the dataset.
        """
        with open(self.dataset_path, "r") as f:
            return json.load(f)

    def _parse_melody(self, melody_str):
        """
        Parses a single melody string into a list of notes.

        Parameters:
            melody_str (str): A string representation of a melody.

        Returns:
            list: A list of notes extracted from the melody string.
        """
        return melody_str.split(", ")

    def _tokenize_and_encode_melodies(self, melodies):
        """
        Tokenizes and encodes a list of melodies.

        Parameters:
            melodies (list): A list of melodies to be tokenized and encoded.

        Returns:
            list: A list of tokenized and encoded melodies.
        """
        unique_tokens = Counter(
            token for melody in melodies for token in melody)
        self.tokenizer = {token: idx + 1 for idx,
                          (token, _) in enumerate(unique_tokens.items())}
        tokenized_melodies = [[self.tokenizer[note]
                               for note in melody] for melody in melodies]
        return tokenized_melodies

    def _set_max_melody_length(self, melodies):
        """
        Sets the maximum melody length based on the dataset.

        Parameters:
            melodies (list): A list of tokenized melodies.
        """
        self.max_melody_length = max(len(melody) for melody in melodies)

    def _set_number_of_tokens(self):
        """
        Sets the number of tokens based on the tokenizer.
        """
        self.number_of_tokens = len(self.tokenizer)

    def _create_sequence_pairs(self, melodies):
        """
        Creates input-target pairs from tokenized melodies.

        Parameters:
            melodies (list): A list of tokenized melodies.

        Returns:
            tuple: Two numpy arrays representing input sequences and target sequences.
        """
        input_sequences, target_sequences = [], []
        for melody in melodies:
            for i in range(1, len(melody)):
                input_seq = melody[:i]
                target_seq = melody[1: i + 1]  # Shifted by one time step
                padded_input_seq = self._pad_sequence(input_seq)
                padded_target_seq = self._pad_sequence(target_seq)
                input_sequences.append(padded_input_seq)
                target_sequences.append(padded_target_seq)
        return np.array(input_sequences), np.array(target_sequences)

    def _pad_sequence(self, sequence):
        """
        Pads a sequence to the maximum sequence length.

        Parameters:
            sequence (list): The sequence to be padded.

        Returns:
            list: The padded sequence.
        """
        return sequence + [0] * (self.max_melody_length - len(sequence))

    def _convert_to_dataloader(self, input_sequences, target_sequences):
        """
        Converts input and target sequences to a PyTorch DataLoader.

        Parameters:
            input_sequences (list): Input sequences for the model.
            target_sequences (list): Target sequences for the model.

        Returns:
            DataLoader: A DataLoader containing input-target pairs.
        """
        class MelodyDataset(Dataset):
            def __init__(self, inputs, targets):
                self.inputs = inputs
                self.targets = targets

            def __len__(self):
                return len(self.inputs)

            def __getitem__(self, idx):
                return torch.tensor(self.inputs[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)

        dataset = MelodyDataset(input_sequences, target_sequences)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


if __name__ == "__main__":
    # Usage example
    preprocessor = MelodyPreprocessor("dataset.json", batch_size=32)
    preprocessor.create_training_dataset()
    training_dataloader = preprocessor.create_training_dataset()
    iterator = iter(training_dataloader)
    first_batch = next(iterator)
    print(first_batch[0][0])
    print(len(first_batch[0][0]))
