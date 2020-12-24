""" 
This module prepares midi file data and feeds it to the neural
network for training     
"""

import glob
import pickle
import numpy
import tensorflow.keras as keras

from music21 import converter, midi, instrument, note, chord

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint

def train_network(folder_name, save_as_filename, 
                  seq_len= 100,
                  LSTM_node_count= 512, 
                  Dropout_count= 0.3,                  
                  epoch= 1, batchsize= 128):
    """ 
    Trains a Neural Network to generate music. 
    `Folder_name` = Folder containing MIDI files you want to train. 
    `Save_as_filename` = Name of output file to be later used to generate MIDI. 
    `Epoch` = Number of times you want the computer to train. 
    `Batchsize` = Number of training examples utilized per epoch. 
    """
    ######################################################
    notes = get_notes(folder_name, save_as_filename)
    
    # get amount of pitch names
    n_vocab = len(set(notes))
    
    # Preparing model
    network_input, network_output = prepare_sequences(notes, n_vocab, seq_len)
    model = create_network(network_input, n_vocab, LSTM_node_count, Dropout_count)
    
    # Training model
    train(model, network_input, network_output, epoch, batchsize)
    #########################################################

def open_midi(midi_path, remove_drums):
    """
    Reads MIDI files into a stream format. 
    There is an option to remove drums if their inputs may tamper with chord analysis.
    """
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    if (remove_drums):
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]
                # Since drums are traditionally located in MIDI channel 10. 
                
    return midi.translate.midiFileToStream(mf)


def get_notes(folder_name, save_as_filename):
    """ 
    Get all the notes and chords from the midi files in the ./midi_songs directory 
    """
    notes = []
        # I might want to add a new line that simplifies chords so that the trained
        # weights are more generalized and can be used for more music during the prediction phase. 
            # Alternatively, complex chords are necessary for new MIDI inputs. 
            # The downside is the weights will be unique to a specific note file.


    for file in glob.glob(folder_name + "/*.mid"):
        #midi = converter.parse(file)
        midi = open_midi(file, True)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('notes_data/' + save_as_filename, 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab, seq_len):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = seq_len

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
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

def create_network(network_input, n_vocab, LSTM_node_count, Dropout_count):
    """ create the structure of the neural network """
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

    return model

def train(model, network_input, network_output, epoch, batchsize):
    """ train the neural network """
    filepath = "trained_weights/" + "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only= True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, 
              network_output, 
              epochs= epoch,
              batch_size= batchsize, 
              callbacks= callbacks_list)
        









