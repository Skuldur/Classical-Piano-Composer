""" 
This module generates notes for a midi file using the
trained neural network 
"""

import pickle
import numpy
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Activation

def generate_midi(test_output_name, notes_file, weight_file, 
                  seq_len= 100, 
                  LSTM_node_count= 512, Dropout_count= 0.3, 
                  note_count= 50,
                  offset_count= 0.5):
    """ 
    Generates a piano midi file.
    
    `notes_file` refers to the file generated using `train_network()`.
    
    `weight_file` refers to the file generated using `train_network()`.
    
    `note_count` = Number of notes you want to generate. 50 by default.
    
    `seq_len` = Number of notes in sequence to sample. 100 by default.
    
    `LSTM_node_count` = Number of nodes to be used in LSTM model per layer. 512 node by default.
    
    `Dropout_count` = Dropout parameter for LSTM mdoel. Accepted inputs from 0-1. Default is 0.3.
    
    `offset_count` = The lower the offset, the "faster" the tempo. Default is 0.5.
    """
    ############################################   
    #load the notes used to train the model
    with open('notes_data/'+ notes_file, 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(set(notes))

    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab, seq_len)
    model = create_network(normalized_input, n_vocab, weight_file, LSTM_node_count, Dropout_count)
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab, note_count)
    
    create_midi(prediction_output, test_output_name, offset_count)
    ##############################################
    
def prepare_sequences(notes, pitchnames, n_vocab, seq_len):
    """ Prepare the sequences used by the Neural Network """
    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = seq_len
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)

def create_network(network_input, n_vocab, weight_file, LSTM_node_count, Dropout_count):
    """ 
    Create the structure of the neural network. 
    `weight_file` refers to the file generated using `train_network()`
    """
    model = Sequential()
    model.add(LSTM(
        LSTM_node_count,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout= Dropout_count,
        return_sequences=True
    ))
    model.add(LSTM(
        LSTM_node_count, 
        return_sequences=True, 
        recurrent_dropout= Dropout_count,))
    model.add(LSTM(LSTM_node_count))
    model.add(BatchNorm())
    model.add(Dropout(Dropout_count))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(Dropout_count))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Load the weights to each node
    model.load_weights("trained_weights/" + weight_file) 

    return model

def generate_notes(model, network_input, pitchnames, n_vocab, note_count):
    """ 
    Generate notes from the neural network based on a sequence of notes 
    """
    # pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # generate N notes
    for note_index in range(note_count):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output, test_output_name, offset_count):
    """ 
    convert the output from the prediction to notes and create a midi file
    from the notes.
    `offset_count` = the note duration before the next one. The lower the offset,
    the "faster" the melody.
    """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += offset_count

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp= 'generated_songs/' + test_output_name + '.mid')
    

