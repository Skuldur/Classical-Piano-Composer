""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, CuDNNLSTM, Embedding, Input, Dropout, LSTM, concatenate, SpatialDropout1D
from keras.models import Model, model_from_json
import math
from keras.utils import Sequence, get_file

B_NOTE = '2'
I_NOTE = '1'
REST = '0'

def train_network():
    """ Train a Neural Network to generate music """
    notes, durations = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))
    n_durations = len(set(durations))
    print('The vocabulary is', n_vocab)
    print('The durations vocabulary is', n_durations)

    notes_input, notes_output = prepare_sequences(notes, n_vocab)
    durations_input, durations_output = prepare_sequences(durations, n_durations)

    model = create_network(n_vocab, set(notes), set(durations))

    #train(model, notes_input, notes_output, durations_input, durations_output)

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []
    durations = []

    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)

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
                #print(element.name, element.duration.quarterLength, element.offset)
                durations.append(str(element.duration.quarterLength))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                #print(element.name, element.duration.quarterLength, element.offset)
                durations.append(str(element.duration.quarterLength))


    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    with open('data/durations', 'wb') as filepath:
        pickle.dump(durations, filepath)

    return notes, durations

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

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

def create_network(n_vocab, pitchnames, unique_durations):
    """ create the structure of the neural network """
    notes_in = Input(shape=(None,))
    emb_notes = Embedding(input_dim=len(pitchnames)+1, output_dim=32)(notes_in)

    duration_in = Input(shape=(None,))
    emb_dur = Embedding(input_dim=len(unique_durations)+1, output_dim=10)(duration_in)

    x = concatenate([emb_notes, emb_dur])
    x = SpatialDropout1D(0.3)(x)

    x = LSTM(
        512,
        return_sequences=True,
        recurrent_dropout=0.3,
    )(x)
    x = LSTM(
        512,
        return_sequences=True,
        recurrent_dropout=0.3,
    )(x)
    x = LSTM(
        512,
    )(x)
    x = Dense(512)(x)
    notes_out = Dense(n_vocab, activation='softmax', name='notes')(x)
    duration_out = Dense(len(unique_durations), activation='softmax', name='duration')(x)


    model = Model(inputs=[notes_in, duration_in], outputs=[notes_out, duration_out])
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[1.0, 0.5], optimizer='rmsprop')

    return model

def train(model, notes_input, notes_output, durations_input, durations_output):
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

    model.load_weights('weights-improvement-100-0.4858-bigger.hdf5')

    train_seq = TrainSequence(notes_input, notes_output, durations_input, durations_output, 128)

    model.fit_generator(
        generator=train_seq,
        epochs=400,
        verbose=1,
        shuffle=False,
        callbacks=callbacks_list,
        initial_epoch=100
    )



class TrainSequence(Sequence):

    def __init__(self, notes_input, notes_output, durations_input, durations_output, batch_size=1, preprocess=None):
        self.notes_in = notes_input
        self.notes_out = notes_output
        self.durations_in = durations_input
        self.durations_out = durations_output
        self.batch_size = batch_size

    def __getitem__(self, idx):
        batch_notes_in = self.notes_in[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_durations_in = self.durations_in[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_notes_out = self.notes_out[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_durations_out = self.durations_out[idx * self.batch_size: (idx + 1) * self.batch_size]

        return [numpy.asarray(batch_notes_in), numpy.asarray(batch_durations_in)], [numpy.asarray(batch_notes_out), numpy.asarray(batch_durations_out)]

    def __len__(self):
        return math.ceil(len(self.notes_in) / self.batch_size)



if __name__ == '__main__':
    train_network()
