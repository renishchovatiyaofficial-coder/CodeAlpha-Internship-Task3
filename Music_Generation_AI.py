"""
===========================================================
TASK 3: MUSIC GENERATION WITH AI
===========================================================

Project Name:
Music Generation with AI

Language:
Python Only

Description:
This project demonstrates AI-based music generation using
a lightweight Recurrent Neural Network (RNN) implemented
entirely with Python's standard library.

The program:
1. Uses built-in melody training data.
2. Converts melodies into numerical note sequences.
3. Trains a simple RNN using gradient descent.
4. Learns patterns between musical notes.
5. Generates a new melody.
6. Converts the generated melody into a MIDI file.
7. Saves the MIDI file to the current folder.

Requirements:
Python 3.10+
No external Python libraries are required.

===========================================================
"""

import math
import random
import struct
import os


# =========================================================
# CONFIGURATION
# =========================================================

EPOCHS = 300
LEARNING_RATE = 0.03
HIDDEN_SIZE = 24
TEMPERATURE = 0.8
GENERATED_LENGTH = 32

OUTPUT_FILE = "AI_Generated_Music.mid"

random.seed(42)


# =========================================================
# BUILT-IN TRAINING DATA
# =========================================================
#
# Notes are represented using MIDI note numbers.
#
# C4  = 60
# D4  = 62
# E4  = 64
# F4  = 65
# G4  = 67
# A4  = 69
# B4  = 71
# C5  = 72
#
# These melodies provide a small classical-style dataset.
# =========================================================

TRAINING_MELODIES = [

    # Melody 1 - C Major
    [60, 62, 64, 65, 67, 67, 65, 64,
     62, 62, 60, 60, 62, 64, 65, 67],

    # Melody 2 - Ascending and descending
    [60, 62, 64, 65, 67, 69, 71, 72,
     71, 69, 67, 65, 64, 62, 60, 60],

    # Melody 3
    [64, 64, 67, 67, 69, 69, 67, 67,
     65, 65, 64, 64, 62, 62, 60, 60],

    # Melody 4
    [60, 64, 67, 72, 67, 64, 60, 64,
     62, 65, 69, 74, 69, 65, 62, 60],

    # Melody 5
    [67, 69, 71, 72, 71, 69, 67, 65,
     64, 65, 67, 69, 67, 65, 64, 62],

    # Melody 6
    [60, 60, 67, 67, 69, 69, 67, 67,
     65, 65, 64, 64, 62, 62, 60, 60],

    # Melody 7
    [72, 71, 69, 67, 65, 64, 62, 60,
     62, 64, 65, 67, 69, 71, 72, 72],

    # Melody 8
    [60, 62, 67, 65, 64, 62, 60, 60,
     67, 69, 72, 71, 69, 67, 65, 64]
]


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def softmax(values):
    """
    Convert a list of values into probabilities.
    """
    maximum = max(values)

    exponentials = [
        math.exp(value - maximum)
        for value in values
    ]

    total = sum(exponentials)

    if total == 0:
        return [1.0 / len(values)] * len(values)

    return [
        value / total
        for value in exponentials
    ]


def random_matrix(rows, columns, scale=0.1):
    """
    Create a random matrix using normal Python lists.
    """
    return [
        [
            random.uniform(-scale, scale)
            for _ in range(columns)
        ]
        for _ in range(rows)
    ]


def random_vector(size, scale=0.1):
    """
    Create a random vector.
    """
    return [
        random.uniform(-scale, scale)
        for _ in range(size)
    ]


def zeros_vector(size):
    """
    Create a vector filled with zeros.
    """
    return [0.0] * size


def tanh_derivative(value):
    """
    Derivative of tanh.
    """
    return 1.0 - value * value


# =========================================================
# PURE PYTHON RNN
# =========================================================

class SimpleRNN:
    """
    A small recurrent neural network implemented from scratch
    using Python lists and math functions.

    Architecture:

        Input
          |
          v
       Hidden RNN
          |
          v
       Output
          |
          v
       Softmax
          |
          v
     Next Note

    This avoids external machine-learning libraries.
    """

    def __init__(self, input_size, hidden_size, output_size):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Input -> Hidden weights
        self.Wxh = random_matrix(
            hidden_size,
            input_size,
            0.15
        )

        # Hidden -> Hidden weights
        self.Whh = random_matrix(
            hidden_size,
            hidden_size,
            0.15
        )

        # Hidden -> Output weights
        self.Why = random_matrix(
            output_size,
            hidden_size,
            0.15
        )

        # Biases
        self.bh = random_vector(hidden_size, 0.05)
        self.by = random_vector(output_size, 0.05)

    # -----------------------------------------------------
    # One-hot encoding
    # -----------------------------------------------------

    def one_hot(self, index):
        vector = [0.0] * self.input_size

        if 0 <= index < self.input_size:
            vector[index] = 1.0

        return vector

    # -----------------------------------------------------
    # Forward pass
    # -----------------------------------------------------

    def forward(self, sequence):

        hidden_states = []
        outputs = []
        probabilities = []

        previous_hidden = zeros_vector(self.hidden_size)

        for note_index in sequence:

            x = self.one_hot(note_index)

            hidden = []

            for h in range(self.hidden_size):

                value = self.bh[h]

                # Input contribution
                for i in range(self.input_size):
                    value += self.Wxh[h][i] * x[i]

                # Previous hidden contribution
                for j in range(self.hidden_size):
                    value += self.Whh[h][j] * previous_hidden[j]

                hidden.append(math.tanh(value))

            output = []

            for o in range(self.output_size):

                value = self.by[o]

                for h in range(self.hidden_size):
                    value += self.Why[o][h] * hidden[h]

                output.append(value)

            probability = softmax(output)

            hidden_states.append(hidden)
            outputs.append(output)
            probabilities.append(probability)

            previous_hidden = hidden

        return hidden_states, outputs, probabilities

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    def train(self, input_sequence, target_sequence):

        hidden_states, outputs, probabilities = self.forward(
            input_sequence
        )

        loss = 0.0

        # Calculate loss
        for t in range(len(target_sequence)):

            target = target_sequence[t]

            probability = probabilities[t][target]

            probability = max(probability, 1e-12)

            loss -= math.log(probability)

        # Gradients
        dWxh = [
            [0.0] * self.input_size
            for _ in range(self.hidden_size)
        ]

        dWhh = [
            [0.0] * self.hidden_size
            for _ in range(self.hidden_size)
        ]

        dWhy = [
            [0.0] * self.hidden_size
            for _ in range(self.output_size)
        ]

        dbh = [0.0] * self.hidden_size
        dby = [0.0] * self.output_size

        dh_next = [0.0] * self.hidden_size

        # Backpropagation Through Time
        for t in reversed(range(len(target_sequence))):

            target = target_sequence[t]

            probability = probabilities[t]

            # Output gradient
            dy = probability[:]

            dy[target] -= 1.0

            # Hidden state
            hidden = hidden_states[t]

            # Previous hidden state
            if t > 0:
                previous_hidden = hidden_states[t - 1]
            else:
                previous_hidden = [
                    0.0
                ] * self.hidden_size

            # Gradient Hidden -> Output
            for o in range(self.output_size):

                for h in range(self.hidden_size):

                    dWhy[o][h] += (
                        dy[o] * hidden[h]
                    )

            # Output bias
            for o in range(self.output_size):
                dby[o] += dy[o]

            # Hidden gradient
            dh = [0.0] * self.hidden_size

            for h in range(self.hidden_size):

                value = dh_next[h]

                for o in range(self.output_size):

                    value += (
                        self.Why[o][h] * dy[o]
                    )

                dh[h] = (
                    value *
                    tanh_derivative(hidden[h])
                )

            # Hidden gradients
            for h in range(self.hidden_size):

                dbh[h] += dh[h]

                # Input
                input_index = input_sequence[t]

                dWxh[h][input_index] += dh[h]

                # Previous hidden
                for j in range(self.hidden_size):

                    dWhh[h][j] += (
                        dh[h] * previous_hidden[j]
                    )

            # Gradient for next time step
            dh_next = [0.0] * self.hidden_size

            for j in range(self.hidden_size):

                for h in range(self.hidden_size):

                    dh_next[j] += (
                        self.Whh[h][j] * dh[h]
                    )

        # Gradient clipping
        gradients = [
            dWxh,
            dWhh,
            dWhy,
            dbh,
            dby
        ]

        for matrix in gradients:

            if isinstance(matrix[0], list):

                for row in matrix:

                    for i in range(len(row)):

                        row[i] = max(
                            -5.0,
                            min(5.0, row[i])
                        )

            else:

                for i in range(len(matrix)):

                    matrix[i] = max(
                        -5.0,
                        min(5.0, matrix[i])
                    )

        # Update Wxh
        for h in range(self.hidden_size):

            for i in range(self.input_size):

                self.Wxh[h][i] -= (
                    LEARNING_RATE *
                    dWxh[h][i]
                )

        # Update Whh
        for h in range(self.hidden_size):

            for j in range(self.hidden_size):

                self.Whh[h][j] -= (
                    LEARNING_RATE *
                    dWhh[h][j]
                )

        # Update Why
        for o in range(self.output_size):

            for h in range(self.hidden_size):

                self.Why[o][h] -= (
                    LEARNING_RATE *
                    dWhy[o][h]
                )

        # Update biases
        for h in range(self.hidden_size):

            self.bh[h] -= (
                LEARNING_RATE *
                dbh[h]
            )

        for o in range(self.output_size):

            self.by[o] -= (
                LEARNING_RATE *
                dby[o]
            )

        return loss

    # -----------------------------------------------------
    # Predict next note
    # -----------------------------------------------------

    def predict_next(self, sequence, temperature=1.0):

        hidden_states, outputs, probabilities = (
            self.forward(sequence)
        )

        last_probability = probabilities[-1]

        # Temperature changes randomness.
        adjusted = []

        for probability in last_probability:

            probability = max(
                probability,
                1e-12
            )

            adjusted.append(
                math.log(probability) /
                max(temperature, 0.05)
            )

        probabilities = softmax(adjusted)

        return probabilities


# =========================================================
# DATASET PREPROCESSING
# =========================================================

def prepare_dataset():

    all_notes = []

    for melody in TRAINING_MELODIES:

        for note in melody:

            if note not in all_notes:
                all_notes.append(note)

    all_notes.sort()

    note_to_index = {
        note: index
        for index, note in enumerate(all_notes)
    }

    index_to_note = {
        index: note
        for note, index in note_to_index.items()
    }

    sequences = []

    for melody in TRAINING_MELODIES:

        encoded = [
            note_to_index[note]
            for note in melody
        ]

        sequences.append(encoded)

    return (
        all_notes,
        note_to_index,
        index_to_note,
        sequences
    )


# =========================================================
# TRAINING
# =========================================================

def train_model(model, sequences):

    print("\n" + "=" * 60)
    print("AI MODEL TRAINING")
    print("=" * 60)

    print(f"Training melodies : {len(sequences)}")
    print(f"Epochs            : {EPOCHS}")
    print(f"Hidden neurons    : {HIDDEN_SIZE}")
    print(f"Learning rate     : {LEARNING_RATE}")

    print("\nTraining started...\n")

    for epoch in range(1, EPOCHS + 1):

        total_loss = 0.0

        # Shuffle melodies every epoch
        training_order = list(range(len(sequences)))

        random.shuffle(training_order)

        for index in training_order:

            sequence = sequences[index]

            if len(sequence) < 2:
                continue

            input_sequence = sequence[:-1]
            target_sequence = sequence[1:]

            loss = model.train(
                input_sequence,
                target_sequence
            )

            total_loss += loss

        if (
            epoch == 1
            or epoch % 25 == 0
            or epoch == EPOCHS
        ):

            average_loss = (
                total_loss /
                max(len(sequences), 1)
            )

            print(
                f"Epoch {epoch:3d}/{EPOCHS} "
                f"| Loss: {average_loss:.4f}"
            )

    print("\nTraining completed successfully.")


# =========================================================
# MUSIC GENERATION
# =========================================================

def generate_music(
    model,
    index_to_note,
    seed_sequence,
    length=32,
    temperature=0.8
):

    generated_indices = seed_sequence[:]

    while len(generated_indices) < length:

        probabilities = model.predict_next(
            generated_indices,
            temperature
        )

        # Random selection based on probabilities
        random_value = random.random()

        cumulative = 0.0

        selected_index = len(probabilities) - 1

        for index, probability in enumerate(
            probabilities
        ):

            cumulative += probability

            if random_value <= cumulative:

                selected_index = index
                break

        generated_indices.append(
            selected_index
        )

    generated_notes = [
        index_to_note[index]
        for index in generated_indices
    ]

    return generated_notes


# =========================================================
# MIDI FILE CREATION
# =========================================================
#
# MIDI is a binary format.
#
# This implementation creates a standard MIDI file using
# Python's built-in struct module.
#
# No external MIDI library is required.
# =========================================================

def variable_length(value):

    """
    Convert an integer into MIDI variable-length format.
    """

    value = int(value)

    buffer = value & 0x7F

    result = bytearray()

    while True:

        value >>= 7

        if value:

            buffer <<= 8
            buffer |= (
                (value & 0x7F) | 0x80
            )

        else:
            break

    while True:

        result.append(buffer & 0xFF)

        if buffer & 0x80:

            buffer >>= 8

        else:

            break

    return bytes(result)


def create_midi_file(
    notes,
    filename,
    tempo=120,
    velocity=80
):

    """
    Create a single-track MIDI file.

    Each note has the same duration.
    """

    ticks_per_beat = 480

    # -----------------------------------------------------
    # MIDI header
    # -----------------------------------------------------

    header = (
        b"MThd" +
        struct.pack(
            ">IHHH",
            6,          # Header length
            0,          # Format 0
            1,          # One track
            ticks_per_beat
        )
    )

    track = bytearray()

    # -----------------------------------------------------
    # Tempo event
    # -----------------------------------------------------

    microseconds_per_beat = int(
        60000000 / tempo
    )

    track.extend(
        variable_length(0)
    )

    track.extend(
        b"\xFF\x51\x03"
    )

    track.extend(
        microseconds_per_beat.to_bytes(
            3,
            byteorder="big"
        )
    )

    # -----------------------------------------------------
    # Instrument
    # -----------------------------------------------------
    #
    # Program 0 = Acoustic Grand Piano
    # -----------------------------------------------------

    track.extend(
        variable_length(0)
    )

    track.extend(
        b"\xC0\x00"
    )

    # -----------------------------------------------------
    # Add notes
    # -----------------------------------------------------

    note_duration = ticks_per_beat

    for note in notes:

        # Note ON
        track.extend(
            variable_length(0)
        )

        track.extend(
            bytes([
                0x90,
                max(0, min(127, note)),
                velocity
            ])
        )

        # Note OFF
        track.extend(
            variable_length(note_duration)
        )

        track.extend(
            bytes([
                0x80,
                max(0, min(127, note)),
                0
            ])
        )

    # -----------------------------------------------------
    # End of track
    # -----------------------------------------------------

    track.extend(
        variable_length(0)
    )

    track.extend(
        b"\xFF\x2F\x00"
    )

    # -----------------------------------------------------
    # Track header
    # -----------------------------------------------------

    track_header = (
        b"MTrk" +
        struct.pack(
            ">I",
            len(track)
        )
    )

    # -----------------------------------------------------
    # Write MIDI file
    # -----------------------------------------------------

    with open(filename, "wb") as file:

        file.write(header)
        file.write(track_header)
        file.write(track)


# =========================================================
# NOTE NAME CONVERSION
# =========================================================

NOTE_NAMES = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B"
]


def midi_to_note_name(midi_number):

    """
    Convert MIDI number to readable note name.

    Example:
    60 -> C4
    64 -> E4
    72 -> C5
    """

    octave = (
        midi_number // 12
    ) - 1

    note_name = NOTE_NAMES[
        midi_number % 12
    ]

    return f"{note_name}{octave}"


# =========================================================
# DISPLAY FUNCTIONS
# =========================================================

def display_training_data():

    print("\n" + "=" * 60)
    print("TRAINING DATA")
    print("=" * 60)

    for index, melody in enumerate(
        TRAINING_MELODIES,
        start=1
    ):

        readable_notes = [
            midi_to_note_name(note)
            for note in melody
        ]

        print(
            f"\nMelody {index}:"
        )

        print(
            " ".join(readable_notes)
        )


def display_generated_music(notes):

    print("\n" + "=" * 60)
    print("GENERATED AI MELODY")
    print("=" * 60)

    readable_notes = [
        midi_to_note_name(note)
        for note in notes
    ]

    print(
        "\nNotes:"
    )

    print(
        " - ".join(readable_notes)
    )

    print(
        f"\nTotal notes: {len(notes)}"
    )


# =========================================================
# OPTIONAL MIDI OPENING
# =========================================================

def open_midi_file(filename):

    """
    Try to open the generated MIDI file using the
    operating system's default application.

    This is optional. If the operating system does not
    support opening MIDI automatically, the file is still
    successfully saved.
    """

    try:

        absolute_path = os.path.abspath(
            filename
        )

        if os.name == "nt":

            os.startfile(
                absolute_path
            )

        elif os.name == "posix":

            import subprocess

            subprocess.Popen(
                ["xdg-open", absolute_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        print(
            "\nAttempting to open the MIDI file..."
        )

    except Exception:

        print(
            "\nThe MIDI file was saved successfully."
        )

        print(
            "Open it manually with a MIDI-compatible "
            "music player."
        )


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():

    print("\n" + "=" * 60)
    print("        MUSIC GENERATION WITH AI")
    print("=" * 60)

    print(
        "\nCodeAlpha Internship - TASK 3"
    )

    print(
        "\nPython-only AI music generation project."
    )

    # -----------------------------------------------------
    # Show dataset
    # -----------------------------------------------------

    display_training_data()

    # -----------------------------------------------------
    # Prepare dataset
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("PREPROCESSING DATA")
    print("=" * 60)

    (
        all_notes,
        note_to_index,
        index_to_note,
        sequences
    ) = prepare_dataset()

    print(
        f"\nUnique musical notes: {len(all_notes)}"
    )

    print(
        "Note vocabulary:",
        " ".join(
            midi_to_note_name(note)
            for note in all_notes
        )
    )

    print(
        "\nMelodies converted into numerical sequences."
    )

    # -----------------------------------------------------
    # Create model
    # -----------------------------------------------------

    model = SimpleRNN(
        input_size=len(all_notes),
        hidden_size=HIDDEN_SIZE,
        output_size=len(all_notes)
    )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    train_model(
        model,
        sequences
    )

    # -----------------------------------------------------
    # Select random seed
    # -----------------------------------------------------

    seed_melody = random.choice(
        TRAINING_MELODIES
    )

    seed_notes = seed_melody[:4]

    seed_indices = [
        note_to_index[note]
        for note in seed_notes
    ]

    print("\n" + "=" * 60)
    print("GENERATION")
    print("=" * 60)

    print(
        "\nSeed notes:"
    )

    print(
        " - ".join(
            midi_to_note_name(note)
            for note in seed_notes
        )
    )

    print(
        "\nGenerating a new melody..."
    )

    # -----------------------------------------------------
    # Generate new music
    # -----------------------------------------------------

    generated_notes = generate_music(
        model=model,
        index_to_note=index_to_note,
        seed_sequence=seed_indices,
        length=GENERATED_LENGTH,
        temperature=TEMPERATURE
    )

    # -----------------------------------------------------
    # Display generated music
    # -----------------------------------------------------

    display_generated_music(
        generated_notes
    )

    # -----------------------------------------------------
    # Save MIDI
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("SAVING MIDI")
    print("=" * 60)

    create_midi_file(
        notes=generated_notes,
        filename=OUTPUT_FILE,
        tempo=120,
        velocity=80
    )

    print(
        f"\nMIDI file created successfully:"
    )

    print(
        os.path.abspath(
            OUTPUT_FILE
        )
    )

    print(
        "\nThe generated composition is ready."
    )

    # -----------------------------------------------------
    # Ask user whether to open MIDI
    # -----------------------------------------------------

    choice = input(
        "\nOpen the generated MIDI file now? "
        "(Y/N): "
    ).strip().lower()

    if choice == "y":

        open_midi_file(
            OUTPUT_FILE
        )

    # -----------------------------------------------------
    # Completion
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("TASK 3 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        "\nAI learned musical note patterns and "
        "generated a new melody."
    )

    print(
        f"MIDI Output: {OUTPUT_FILE}"
    )

    print(
        "\nThank you!"
    )


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()


