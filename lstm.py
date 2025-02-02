""" This module prepares midi file data and feeds it to the neural
    network for training """
import numpy as np 
import glob
import pickle
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization as BatchNorm, Input
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes()

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(pitchnames)  # Use pitchnames for vocabulary size

    # Prepare sequences
    normalized_input, network_output = prepare_sequences(notes, pitchnames, n_vocab)

    # Pass the sequence length to create_network
    sequence_length = normalized_input.shape[1]
    model = create_network(n_vocab, sequence_length)

    # Train the model
    train(model, normalized_input, network_output)  # Call the train function

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)
        print("Parsing %s" % file)

        notes_to_parse = None

        try:  # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape the input into a format compatible with LSTM layers
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))  # Ensure this is 3D
    normalized_input = normalized_input / float(n_vocab)

    # Convert output to categorical
    network_output = to_categorical(output, num_classes=n_vocab)

    return normalized_input, network_output  # Return two values

def create_network(n_vocab, sequence_length):
    """ Create the structure of the neural network """
    model = Sequential()
    model.add(Input(shape=(sequence_length, 1)))  # Use Input layer for shape
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Load the weights to each node if they exist
    try:
        model.load_weights('weights.hdf5')
    except Exception as e:
        print("Could not load weights:", e)

    return model

def train(model, network_input, network_output):
    """ Train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.keras"  # Change to .keras
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=1,  # Set to 1 to see messages when saving the model
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=200, batch_size=128, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network()