""" This module prepares midi file data and feeds it to the neural
    network for training """
import os
import glob
import pickle
import numpy
import tensorflow as tf
from music21 import converter, instrument, note, chord
from keras.models import load_model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from network import create_network


def train_network():
    """Train a Neural Network to generate music"""
    notes = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    checkpoints = ["checkpoints/" + name for name in os.listdir("checkpoints/")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"*** Restoring from the lastest checkpoint: {latest_checkpoint} ***")
        model = load_model(latest_checkpoint)
    else:
        model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)


def get_notes():
    """Get all the notes and chords from the midi files in the ./midi_songs directory"""
    notes = []

    for file in glob.glob("midi_songs/*.midi"):
        midi = converter.parse(file)

        print(f"Parsing {file}")

        try:  # file has instrument parts
            instrument_stream = instrument.partitionByInstrument(midi)
            notes_to_parse = instrument_stream.parts[0].recurse()
        except:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(n) for n in element.normalOrder))

    with open("data/notes", "wb") as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, n_vocab):
    """Prepare the sequences used by the Neural Network"""
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i : i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)


def train(model, network_input, network_output):
    """train the neural network"""
    filepath = "checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor="loss", verbose=0, save_best_only=True, mode="min"
    )
    callbacks_list = [checkpoint]

    model.fit(
        network_input,
        network_output,
        epochs=200,
        batch_size=128,
        callbacks=callbacks_list,
    )


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    train_network()
