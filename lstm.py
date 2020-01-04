""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
import math
from music21 import converter, instrument, note, chord
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, CuDNNLSTM, Embedding, Input, Dropout, LSTM, concatenate, SpatialDropout1D, Reshape, Activation, Permute, Multiply, Lambda, RepeatVector
from keras.models import Model, model_from_json
import math
from keras import backend as K
from keras.utils import Sequence, get_file
from keras.regularizers import l1

B_NOTE = '2'
I_NOTE = '1'
REST = '0'

SEQUENCE_LENGTH = 50

def train_network():
    """ Train a Neural Network to generate music """
    notes, durations, offsets = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))
    n_durations = len(set(durations))
    n_offsets = len(set(offsets))
    print('The vocabulary is', n_vocab)
    print('The durations vocabulary is', n_durations)
    print('The durations vocabulary is', n_offsets)
    print(set(durations))
    print(set(offsets))

    notes_input, notes_output = prepare_sequences(notes, n_vocab)
    durations_input, durations_output = prepare_sequences(durations, n_durations)
    offsets_input, offsets_output = prepare_sequences(offsets, n_offsets)

    model = create_network(n_vocab, set(notes), set(durations), set(offsets))

    train(model, notes_input, notes_output, durations_input, durations_output, offsets_input, offsets_output)

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []
    durations = []
    offsets=[]



    for file in glob.glob("midi_songs/*.mid"):
        print("Parsing %s" % file, len(notes))
        midi = converter.parse(file)


        notes.append('START')
        durations.append(0.)
        offsets.append(0.)


        notes_to_parse = None
        prev_offset = 0.

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
            if 'Barfuss_am_Klavier' in file:
                notes_to_parse = s2.parts[1].recurse()
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
                #print(element.name, element.duration.quarterLength, element.offset)
                durations.append(float(element.duration.quarterLength))
                offsets.append(float(element.offset) - float(prev_offset))
                prev_offset = element.offset
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                #print(element.name, element.duration.quarterLength, element.offset)
                durations.append(float(element.duration.quarterLength))
                offsets.append(float(element.offset) - float(prev_offset))
                prev_offset = element.offset

        notes.append('END')
        durations.append(float('inf'))
        offsets.append(float('inf'))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    with open('data/durations', 'wb') as filepath:
        pickle.dump(durations, filepath)

    with open('data/offsets', 'wb') as filepath:
        pickle.dump(offsets, filepath)

    return notes, durations, offsets

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = SEQUENCE_LENGTH

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
    #network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    #network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def create_network(n_vocab, pitchnames, unique_durations, unique_offsets):
    """ create the structure of the neural network """
    notes_in = Input(shape=(None,))
    emb_notes = Embedding(input_dim=len(pitchnames), output_dim=32)(notes_in)

    duration_in = Input(shape=(None,))
    emb_dur = Embedding(input_dim=len(unique_durations), output_dim=32)(duration_in)

    offsets_in = Input(shape=(None,))
    emb_off = Embedding(input_dim=len(unique_offsets), output_dim=32)(offsets_in)

    x = concatenate([emb_notes, emb_dur, emb_off])
    #x = SpatialDropout1D(0.2)(x)


    x = LSTM(
        256,
        return_sequences=True,
        #recurrent_dropout=0.2,
    )(x)    
    x = LSTM(
        256,
        return_sequences=True,
        #recurrent_dropout=0.2,
    )(x)
    """x = LSTM(
        256,
        recurrent_dropout=0.2
    )(x)"""

    e = Dense(1, activation='tanh')(x)
    e = Reshape([-1])(e)
    alpha = Activation('softmax')(e)
    c = Permute([2, 1])(RepeatVector(256)(alpha))
    c = Multiply()([x, c])
    c = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(256,))(c)


    #x = Dense(512)(x)
    notes_out = Dense(n_vocab, activation='softmax', name='notes')(c)
    duration_out = Dense(len(unique_durations), activation='softmax', name='duration')(c)
    offset_out = Dense(len(unique_offsets), activation='softmax', name='offset')(c)


    model = Model(inputs=[notes_in, duration_in, offsets_in], outputs=[notes_out, duration_out, offset_out])
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[1, 0.5, 0.5], optimizer='rmsprop')

    return model

def train(model, notes_input, notes_output, durations_input, durations_output, offset_input, offset_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    #model.load_weights('weights-improvement-100-0.4858-bigger.hdf5')

    train_seq = TrainSequence(
        notes_input, notes_output, 
        durations_input, durations_output,
        offset_input, offset_output, 
        128)

    model.fit_generator(
        generator=train_seq,
        epochs=400,
        verbose=1,
        shuffle=False,
        callbacks=callbacks_list,
        #initial_epoch=100
    )



class TrainSequence(Sequence):

    def __init__(self, 
        notes_input, notes_output, 
        durations_input, durations_output, 
        offset_input, offset_output,  
        batch_size=1, preprocess=None):
        self.notes_in = notes_input
        self.notes_out = notes_output
        self.durations_in = durations_input
        self.durations_out = durations_output
        self.offsets_in = offset_input
        self.offsets_out = offset_output
        self.batch_size = batch_size

    def __getitem__(self, idx):
        batch_notes_in = self.notes_in[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_durations_in = self.durations_in[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_offsets_in = self.offsets_in[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_notes_out = self.notes_out[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_durations_out = self.durations_out[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_offsets_out = self.offsets_out[idx * self.batch_size: (idx + 1) * self.batch_size]

        return [numpy.asarray(batch_notes_in), numpy.asarray(batch_durations_in), numpy.asarray(batch_offsets_in)], [numpy.asarray(batch_notes_out), numpy.asarray(batch_durations_out), numpy.asarray(batch_offsets_out)]

    def __len__(self):
        return math.ceil(len(self.notes_in) / self.batch_size)



if __name__ == '__main__':
    train_network()
