import torch
import torch.nn.functional as F


class MelodyGenerator:
    """
    Class to generate melodies using a trained Transformer model.

    This class encapsulates the inference logic for generating melodies
    based on a starting sequence.
    """

    def __init__(self, transformer, tokenizer, max_length=50):
        """
        Initializes the MelodyGenerator.

        Parameters:
            transformer (Transformer): The trained Transformer model.
            tokenizer (Tokenizer): Tokenizer used for encoding melodies.
            max_length (int): Maximum length of the generated melodies.
        """
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.max_length = max_length

    def generate(self, start_sequence):
        """
        Generates a melody based on a starting sequence.

        Parameters:
            start_sequence (list of str): The starting sequence of the melody.

        Returns:
            str: The generated melody.
        """
        input_tensor = self._get_input_tensor(start_sequence)

        num_notes_to_generate = self.max_length - input_tensor.size(1)

        for _ in range(num_notes_to_generate):
            # change the None
            predictions = self.transformer(
                input_tensor, input_tensor, None, None, None, None)
            predicted_note = self._get_note_with_highest_score(predictions)
            input_tensor = self._append_predicted_note(
                input_tensor, predicted_note)

        generated_melody = self._decode_generated_sequence(input_tensor)

        return generated_melody

    def _get_input_tensor(self, start_sequence):
        """
        Gets the input tensor for the Transformer model.

        Parameters:
            start_sequence (list of str): The starting sequence of the melody.

        Returns:
            input_tensor (torch.Tensor): The input tensor for the model.
        """
        # input_sequence = self.tokenizer.texts_to_sequences([start_sequence])
        input_sequence = [[self.tokenizer[note]
                           for note in melody] for melody in [start_sequence]]
        input_tensor = torch.tensor(input_sequence, dtype=torch.long)
        return input_tensor

    def _get_note_with_highest_score(self, predictions):
        """
        Gets the note with the highest score from the predictions.

        Parameters:
            predictions (torch.Tensor): The predictions from the model.

        Returns:
            predicted_note (int): The index of the predicted note.
        """
        latest_predictions = predictions[:, -1, :]
        predicted_note_index = torch.argmax(latest_predictions, dim=1)
        predicted_note = predicted_note_index.item()
        return predicted_note

    def _append_predicted_note(self, input_tensor, predicted_note):
        """
        Appends the predicted note to the input tensor.

        Parameters:
            input_tensor (torch.Tensor): The input tensor for the model.

        Returns:
            torch.Tensor: The input tensor with the predicted note appended.
        """
        predicted_note_tensor = torch.tensor(
            [[predicted_note]], dtype=torch.long)
        return torch.cat([input_tensor, predicted_note_tensor], dim=1)

    def _decode_generated_sequence(self, generated_sequence):
        """
        Decodes the generated sequence of notes.

        Parameters:
            generated_sequence (torch.Tensor): Tensor with note indexes generated.

        Returns:
            generated_melody (str): The decoded sequence of notes.
        """
        generated_sequence_array = generated_sequence.squeeze().tolist()
        # generated_melody = self.tokenizer.sequences_to_texts(
        #    generated_sequence_array)[0]
        generated_melody = self.sequences_to_texts(generated_sequence_array)
        return generated_melody

    def sequences_to_texts(self, sequences):
        text = []
        for sequence in sequences:
            for key, value in self.tokenizer.items():
                if value == sequence:
                    text.append(key)
        return text


if __name__ == "__main__":
    from transformer import Transformer
    from melodypreprocessor import MelodyPreprocessor
    from torch.utils.data import DataLoader

    EPOCHS = 10
    BATCH_SIZE = 32
    DATA_PATH = "dataset.json"
    MAX_POSITIONS_IN_POSITIONAL_ENCODING = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess dataset
    melody_preprocessor = MelodyPreprocessor(DATA_PATH, batch_size=BATCH_SIZE)
    train_dataset = DataLoader(melody_preprocessor.create_training_dataset(
    ), batch_size=BATCH_SIZE, shuffle=True)
    vocab_size = melody_preprocessor.number_of_tokens_with_padding

    # Initialize Transformer model
    transformer_model = Transformer(
        num_layers=2,
        d_model=64,
        num_heads=2,
        d_feedforward=128,
        input_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        max_num_positions_in_pe_encoder=100,
        max_num_positions_in_pe_decoder=100,
        dropout_rate=0.1,
    ).to(device)

    # Generate a melody
    print("Generating a melody...")
    melody_generator = MelodyGenerator(
        transformer_model, melody_preprocessor.tokenizer)
    start_sequence = ["C4-1.0", "D4-1.0", "E4-1.0", "C4-1.0"]
    new_melody = melody_generator.generate(start_sequence)
    print(f"Generated melody: {new_melody}")
