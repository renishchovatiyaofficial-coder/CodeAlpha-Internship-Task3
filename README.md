# CodeAlpha-Internship3

# Music Generation with AI

## CodeAlpha Artificial Intelligence Internship - Task 3

### Project Overview

Music Generation with AI is a Python-based artificial intelligence project that demonstrates how a recurrent neural network can learn patterns from musical note sequences and generate a new melody.

The project uses a lightweight RNN implemented from scratch using Python. It does not require external machine learning or music-processing libraries.

The trained model analyzes built-in classical-style melodies, learns relationships between consecutive notes, and generates a new sequence of musical notes. The generated sequence is then converted into a standard MIDI file.

---

## Project Information

**Project Name:** Music Generation with AI

**Task:** Task 3

**Internship:** CodeAlpha Artificial Intelligence Internship

**Programming Language:** Python

**Main File:** `Music_Generation_with_AI.py`

**Output File:** `AI_Generated_Music.mid`

---

## Features

- Built-in musical training dataset
- Musical note sequence preprocessing
- Pure Python recurrent neural network
- Recurrent hidden-state processing
- Backpropagation Through Time
- Gradient descent training
- Softmax-based note prediction
- Temperature-based music generation
- Automatic melody generation
- MIDI file creation
- MIDI file saving
- Optional automatic MIDI opening
- No external machine learning framework required
- Beginner-friendly implementation
- Fully self-contained Python program

---

## Technologies Used

This project is implemented using Python's standard library.

### Python Modules

- `math`
- `random`
- `struct`
- `os`

No external packages are required.

---

## Libraries Not Used

This project intentionally does not use:

- NumPy
- TensorFlow
- Keras
- PyTorch
- music21
- Flask
- JavaScript
- HTML
- CSS

The neural network and MIDI generation logic are implemented directly in Python.

---

## How the Project Works

The project follows several stages.

### 1. Musical Training Data

The program contains a small built-in dataset of classical-style melodies.

The melodies are represented using MIDI note numbers.

For example:

```text
60 = C4
62 = D4
64 = E4
65 = F4
67 = G4
69 = A4
71 = B4
72 = C5
```

The built-in dataset contains multiple melody patterns that the RNN can learn from.

---

### 2. Data Preprocessing

The program identifies all unique musical notes in the dataset.

Each note is assigned an index.

For example:

```text
C4 -> 0
D4 -> 1
E4 -> 2
F4 -> 3
G4 -> 4
A4 -> 5
B4 -> 6
C5 -> 7
```

The melodies are then converted into numerical sequences suitable for the neural network.

---

### 3. RNN Model

The project implements a small recurrent neural network from scratch.

The architecture is:

```text
Input Note
    |
    v
Input-to-Hidden Layer
    |
    v
Recurrent Hidden Layer
    |
    v
Hidden-to-Output Layer
    |
    v
Softmax
    |
    v
Predicted Next Note
```

The recurrent hidden layer allows the model to use information from previous notes when predicting the next note.

---

### 4. Model Training

The model is trained using the musical sequences.

For every sequence, the program uses:

```text
Input:
C4 D4 E4 F4

Target:
D4 E4 F4 G4
```

The model learns to predict the next musical note based on the previous note sequence.

Training uses:

- Forward propagation
- Loss calculation
- Backpropagation Through Time
- Gradient clipping
- Gradient descent
- Weight updates

---

### 5. Music Generation

After training, the program selects a seed sequence from the training dataset.

The RNN predicts the next note.

The predicted note is added to the sequence, and the process continues until the requested melody length is reached.

The generation process can be represented as:

```text
Seed Notes
    |
    v
RNN Prediction
    |
    v
Next Note
    |
    v
Add Note to Sequence
    |
    v
RNN Prediction
    |
    v
Next Note
    |
    v
Continue...
```

---

### 6. Temperature

The project uses a temperature value to control randomness during music generation.

The setting is:

```python
TEMPERATURE = 0.8
```

A lower temperature generally produces more predictable results.

A higher temperature produces more variation.

---

### 7. MIDI Generation

After the new melody is generated, the program converts the notes into a standard MIDI file.

The MIDI file contains:

- MIDI header
- Tempo information
- Piano instrument
- Note-on events
- Note-off events
- Note duration
- MIDI end-of-track event

The generated file is:

```text
AI_Generated_Music.mid
```

---

## Requirements

You only need Python installed on your computer.

Recommended version:

```text
Python 3.10 or newer
```

The project is designed to run with Python 3.13 as well.

No `pip install` command is required.

---

## Project Structure

```text
Music-Generation-with-AI/
│
├── Music_Generation_with_AI.py
├── AI_Generated_Music.mid
└── README.md
```

The MIDI file will be created automatically after running the Python program.

---

## How to Run

### Step 1: Install Python

Download and install Python from the official Python website if Python is not already installed.

Make sure Python is added to the system PATH during installation.

### Step 2: Save the Python File

Save the program as:

```text
Music_Generation_with_AI.py
```

### Step 3: Open the Terminal

Navigate to the folder containing the Python file.

Example:

```text
cd path\to\Music-Generation-with-AI
```

### Step 4: Run the Program

Use:

```text
python Music_Generation_with_AI.py
```

On some Windows systems, you can also use:

```text
py Music_Generation_with_AI.py
```

### Step 5: Wait for Training

The program will preprocess the melodies and train the RNN.

You will see training progress similar to:

```text
Epoch   1/300 | Loss: ...
Epoch  25/300 | Loss: ...
Epoch  50/300 | Loss: ...
Epoch  75/300 | Loss: ...
...
Epoch 300/300 | Loss: ...
```

### Step 6: Generate Music

After training, the program generates a new melody.

The generated notes will be displayed in the terminal.

### Step 7: Save MIDI

The program automatically creates:

```text
AI_Generated_Music.mid
```

You can open this file using a MIDI-compatible music player or digital audio workstation.

---

## Example Output

```text
============================================================
        MUSIC GENERATION WITH AI
============================================================

CodeAlpha Internship - TASK 3

Python-only AI music generation project.

============================================================
PREPROCESSING DATA
============================================================

Unique musical notes: 8

Note vocabulary: C4 D4 E4 F4 G4 A4 B4 C5

Melodies converted into numerical sequences.

============================================================
AI MODEL TRAINING
============================================================

Training melodies : 8
Epochs            : 300
Hidden neurons    : 24
Learning rate     : 0.03

Training started...

Epoch   1/300 | Loss: ...
Epoch  25/300 | Loss: ...
Epoch  50/300 | Loss: ...
Epoch  75/300 | Loss: ...
Epoch 100/300 | Loss: ...
Epoch 125/300 | Loss: ...
Epoch 150/300 | Loss: ...
Epoch 175/300 | Loss: ...
Epoch 200/300 | Loss: ...
Epoch 225/300 | Loss: ...
Epoch 250/300 | Loss: ...
Epoch 275/300 | Loss: ...
Epoch 300/300 | Loss: ...

Training completed successfully.

============================================================
GENERATION
============================================================

Seed notes:
C4 - E4 - G4 - C5

Generating a new melody...

============================================================
GENERATED AI MELODY
============================================================

Notes:
C4 - E4 - G4 - G4 - A4 - G4 - E4 - D4 ...

Total notes: 32

============================================================
SAVING MIDI
============================================================

MIDI file created successfully:
...\AI_Generated_Music.mid

============================================================
TASK 3 COMPLETED SUCCESSFULLY
============================================================

AI learned musical note patterns and generated a new melody.

MIDI Output: AI_Generated_Music.mid
```

The exact generated notes and loss values can vary because the model uses probabilistic generation.

---

## MIDI Output

The generated MIDI file can be opened with any application that supports MIDI playback.

Examples include:

- Windows Media Player versions that support MIDI
- VLC Media Player
- GarageBand
- FL Studio
- Ableton Live
- Other MIDI-compatible applications

The project itself only creates the MIDI file. MIDI playback depends on the software and operating system available on the computer.

---

## RNN Architecture

The model contains three primary stages.

### Input Layer

Receives the encoded musical note.

### Hidden Recurrent Layer

Maintains information from previous notes and learns musical patterns.

The hidden layer contains:

```text
24 neurons
```

### Output Layer

Produces probabilities for the possible next musical notes.

Softmax converts the output values into probabilities.

---

## Training Configuration

The main configuration values are:

```python
EPOCHS = 300
LEARNING_RATE = 0.03
HIDDEN_SIZE = 24
TEMPERATURE = 0.8
GENERATED_LENGTH = 32
```

These values can be changed to experiment with the model.

---

## Advantages

- Completely Python-based
- No external dependencies
- Easy to understand
- Demonstrates fundamental RNN concepts
- Includes actual model training
- Generates original note sequences from learned patterns
- Produces a real MIDI file
- Easy to run on a basic computer
- Suitable for an educational AI internship project

---

## Limitations

This project uses a small built-in dataset for demonstration and educational purposes.

Because the dataset is small, the generated music has limited musical complexity.

A larger real-world music-generation system would require:

- A much larger MIDI dataset
- More musical features
- More training data
- A larger neural network
- Longer training
- More advanced architectures
- Better sequence representation
- More sophisticated MIDI processing

The purpose of this project is to demonstrate the core concept of AI-based sequence learning and music generation in a simple Python-only implementation.

---

## Future Improvements

Possible future improvements include:

1. Train on a larger MIDI dataset.
2. Add multiple musical instruments.
3. Add note duration information.
4. Add velocity information.
5. Add chords.
6. Add multiple tracks.
7. Add different musical genres.
8. Improve the RNN architecture.
9. Add LSTM cells.
10. Implement attention mechanisms.
11. Add automatic WAV/audio rendering.
12. Generate longer compositions.
13. Add melody and harmony generation.
14. Save multiple generated compositions.
15. Build a larger music vocabulary.

---

## Learning Outcomes

This project demonstrates the following concepts:

- Artificial Intelligence
- Machine Learning
- Recurrent Neural Networks
- Sequence Modeling
- Musical Data Representation
- Data Preprocessing
- One-Hot Encoding
- Softmax Probability
- Gradient Descent
- Backpropagation Through Time
- Model Training
- Pattern Recognition
- Generative AI
- MIDI File Structure
- Algorithmic Music Generation

---

## Internship Requirement Mapping

| CodeAlpha Requirement | Project Implementation |
|---|---|
| Collect MIDI music data | Built-in classical-style melody dataset |
| Preprocess note sequences | Musical notes converted into numerical sequences |
| Build deep learning model | Pure Python recurrent neural network |
| Train the model | Gradient descent and Backpropagation Through Time |
| Learn music patterns | RNN learns relationships between sequential notes |
| Generate new music | Probabilistic next-note generation |
| Convert generated sequences to MIDI | Custom MIDI file generator |
| Save generated music | `AI_Generated_Music.mid` |
| Play music | Optional system MIDI opening |

---

## Conclusion

Music Generation with AI demonstrates how a recurrent neural network can learn sequential patterns from musical notes and use those learned patterns to generate a new melody.

The project is intentionally implemented using Python's standard library so that the fundamental concepts of sequence modeling, neural-network training, music generation, and MIDI file creation can be understood without depending on large machine-learning frameworks.

The final result is a generated MIDI composition saved as:

```text
AI_Generated_Music.mid
```

---

## Author

**Renish**

CodeAlpha Artificial Intelligence Internship

Task 3: Music Generation with AI
