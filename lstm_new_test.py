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

from keras.preprocessing.sequence import pad_sequences

START_NOTE = 0

SEQUENCE_LENGTH = 32

def train_network():
    """ Train a Neural Network to generate music """
    notes, durations, offsets, time = get_notes()

    # get amount of pitch names

    unique_notes = get_set_of_values(notes)
    unique_durations = get_set_of_values(durations)
    unique_offsets = get_set_of_values(offsets)
    unique_time = get_set_of_values(time)

    n_vocab = len(unique_notes)
    n_durations = len(unique_durations)
    n_offsets = len(unique_offsets)
    n_time = len(unique_time)
    print('The vocabulary is', n_vocab)
    print('The durations vocabulary is', n_durations)
    print('The durations vocabulary is', n_offsets)
    print(unique_durations)
    print(unique_offsets)

    notes_input, notes_output = prepare_sequences(notes, n_vocab, unique_notes)

    durations_input, durations_output = prepare_sequences(durations, n_durations, unique_durations)
    offsets_input, offsets_output = prepare_sequences(offsets, n_offsets, unique_offsets)
    time_input, time_output = prepare_sequences(time, n_time, unique_time)

    model = create_network(len(notes_output[0]), unique_notes, unique_durations, unique_offsets, unique_time)

    #train(model, notes_input, notes_output, durations_input, durations_output, offsets_input, offsets_output, time_input, time_output)

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []
    durations = []
    offsets=[]
    time = []

    i = 0

    for file in glob.glob("midi_songs/*.mid"):
        print("Parsing %s, - %s" % (file, i))
        i += 1
        midi = converter.parse(file)

        file_notes = []
        file_durations = []
        file_offsets = []
        file_time = []


        file_notes.append('START')
        file_durations.append(0.)
        file_offsets.append(0.)
        file_time.append(128)


        notes_to_parse = None
        prev_offset = 0.

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
            print(len(s2.parts[0].notes))
            length = len(s2.parts[0].notes)
            if 'Barfuss_am_Klavier' in file:
                notes_to_parse = s2.parts[1].recurse()
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
            print(len(notes_to_parse))
            length = len(notes_to_parse)

        time_step = length / 127.
        #print("TIME STEP IS ", time_step)

        count_notes = 0

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                file_notes.append(str(element.pitch))
                file_durations.append(float(element.duration.quarterLength))
                file_offsets.append(float(element.offset) - float(prev_offset))
                prev_offset = element.offset
                file_time.append(127 - math.floor(count_notes / time_step))
                count_notes += 1
            elif isinstance(element, chord.Chord):
                file_notes.append('.'.join(str(n) for n in element.normalOrder))
                file_durations.append(float(element.duration.quarterLength))
                file_offsets.append(float(element.offset) - float(prev_offset))
                prev_offset = element.offset
                file_time.append(127 - math.floor(count_notes / time_step))
                count_notes += 1

        file_notes.append('END')
        file_durations.append(float('inf'))
        file_offsets.append(float('inf'))
        file_time.append(0)

        #print('The actual length', len(file_notes))
        #print(file_time)

        notes.append(file_notes)
        durations.append(file_durations)
        offsets.append(file_offsets)
        time.append(file_time)

    """with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    with open('data/durations', 'wb') as filepath:
        pickle.dump(durations, filepath)

    with open('data/offsets', 'wb') as filepath:
        pickle.dump(offsets, filepath)
    with open('data/time', 'wb') as filepath:
        pickle.dump(time, filepath)"""


    return notes, durations, offsets, time

def prepare_sequences(notes, n_vocab, set_of_notes):
    global START_NOTE
    """ Prepare the sequences used by the Neural Network """
    sequence_length = SEQUENCE_LENGTH

    # get all pitch names
    pitchnames = sorted(set_of_notes)

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    try:
        print('THIS IS START', note_to_int['START'])
        START_NOTE = note_to_int['START']
    except:
        print('duh')

    network_input = []
    network_output = []

    for i, file in enumerate(notes):
        print('File %s starts at %s' % (i, len(network_input)))
        slow = 0
        fast = SEQUENCE_LENGTH

        #for i in range(0, len(notes) - sequence_length, 1):
        while fast < len(file):
            if fast - slow < SEQUENCE_LENGTH:
                sequence_in = file[slow:fast]
                sequence_out = file[fast]
                network_input.append([note_to_int[char] for char in sequence_in])
                network_output.append(note_to_int[sequence_out])
            elif fast - slow == SEQUENCE_LENGTH:
                sequence_in = file[slow:fast]
                sequence_out = file[fast]
                network_input.append([note_to_int[char] for char in sequence_in])
                network_output.append(note_to_int[sequence_out])
                slow += 1
            fast += 1






    # create input sequences and the corresponding outputs
    """for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])"""

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    #network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    #network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    print(len(network_output[0]))

    return (network_input, network_output)

def create_network(note_output, unique_notes, unique_durations, unique_offsets, unique_time):
    """ create the structure of the neural network """
    notes_in = Input(shape=(None,))
    emb_notes = Embedding(input_dim=len(unique_notes), output_dim=32)(notes_in)

    duration_in = Input(shape=(None,))
    emb_dur = Embedding(input_dim=len(unique_durations), output_dim=32)(duration_in)

    offsets_in = Input(shape=(None,))
    emb_off = Embedding(input_dim=len(unique_offsets), output_dim=32)(offsets_in)

    time_in = Input(shape=(None,))
    emb_time = Embedding(input_dim=len(unique_time), output_dim=32)(time_in)

    x = concatenate([emb_notes, emb_dur, emb_off, emb_time])
    #x = SpatialDropout1D(0.2)(x)


    x = LSTM(
        512,
        return_sequences=True,
        #recurrent_dropout=0.2,
    )(x)    
    x = LSTM(
        512,
        return_sequences=True,
        #recurrent_dropout=0.2,
    )(x)
    x = LSTM(
        512,
        return_sequences=True,
        #recurrent_dropout=0.2,
    )(x)
    """x = LSTM(
        512,
        recurrent_dropout=0.2
    )(x)"""

    e = Dense(1, activation='tanh')(x)
    e = Reshape([-1])(e)
    alpha = Activation('softmax')(e)
    c = Permute([2, 1])(RepeatVector(512)(alpha))
    c = Multiply()([x, c])
    c = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(512,))(c)


    #x = Dense(512)(x)
    notes_out = Dense(note_output, activation='softmax', name='notes')(c)
    duration_out = Dense(len(unique_durations), activation='softmax', name='duration')(c)
    offset_out = Dense(len(unique_offsets), activation='softmax', name='offset')(c)
    time_out = Dense(124, activation='softmax', name='time')(c)


    model = Model(inputs=[notes_in, duration_in, offsets_in, time_in], outputs=[notes_out, duration_out, offset_out, time_out])
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[1, 0.5, 0.5, 0.5], optimizer='rmsprop')

    return model

def train(model, notes_input, notes_output, durations_input, durations_output, offset_input, offset_output, time_input, time_output):
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

    print(len(time_output[0]))

    #model.load_weights('weights-improvement-100-0.4858-bigger.hdf5')

    train_seq = TrainSequence(
        notes_input, notes_output, 
        durations_input, durations_output,
        offset_input, offset_output,
        time_input, time_output,
        32)

    model.fit_generator(
        generator=train_seq,
        epochs=400,
        verbose=1,
        shuffle=False,
        callbacks=callbacks_list,
        #initial_epoch=100
    )



def get_set_of_values(matrix):
    s = set()

    for row in matrix:
        for item in row:
            s.add(item)

    return s

class TrainSequence(Sequence):

    def __init__(self, 
        notes_input, notes_output, 
        durations_input, durations_output, 
        offset_input, offset_output, 
        time_input, time_output,
        batch_size=1, preprocess=None):
        self.notes_in = notes_input
        self.notes_out = notes_output
        self.durations_in = durations_input
        self.durations_out = durations_output
        self.offsets_in = offset_input
        self.offsets_out = offset_output
        self.time_in = time_input
        self.time_out = time_output
        self.batch_size = batch_size

    def __getitem__(self, idx):
        batch_notes_in = self.notes_in[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_durations_in = self.durations_in[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_offsets_in = self.offsets_in[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_time_in = self.time_in[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_notes_out = self.notes_out[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_durations_out = self.durations_out[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_offsets_out = self.offsets_out[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_time_out = self.time_out[idx * self.batch_size: (idx + 1) * self.batch_size]

        #print('input')
        batch_input = [
            numpy.asarray(pad_sequences(sequences=batch_notes_in, padding="pre", value=START_NOTE)), 
            numpy.asarray(pad_sequences(sequences=batch_durations_in, padding="pre", value=0.)), 
            numpy.asarray(pad_sequences(sequences=batch_offsets_in, padding="pre", value=0.)),
            numpy.asarray(pad_sequences(sequences=batch_time_in, padding="pre", value=128))
        ]
        #print('output')
        batch_output = [
            numpy.asarray(batch_notes_out), 
            numpy.asarray(batch_durations_out), 
            numpy.asarray(batch_offsets_out),
            numpy.asarray(batch_time_out)
        ]

        return batch_input, batch_output

    def __len__(self):
        return math.ceil(len(self.notes_in) / self.batch_size)



if __name__ == '__main__':
    train_network()
