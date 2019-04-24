""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, CuDNNLSTM, Embedding, Input, Dropout, LSTM
from keras.models import Model, model_from_json
import math
from keras.utils import Sequence, get_file

B_NOTE = '2'
I_NOTE = '1'
REST = '0'

def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))
    print('The vocabulary is', n_vocab)

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab, set(notes))

    train(model, network_input, network_output)

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

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
                #print(str(element.pitch), element.pitch.pitchClass)
                note_class = element.pitch.pitchClass
                for i in range(0, int(float(element.duration.quarterLength)/0.25)):
                    curr_time = [REST]*12

                    if i == 0:
                        curr_time[note_class] = B_NOTE
                    else:
                        curr_time[note_class] = I_NOTE

                    notes.append(curr_time)
                #s.add(element.name)
            elif isinstance(element, chord.Chord):
                #print(str(element.normalOrder), float(element.duration.quarterLength))
                #notes.append('.'.join(str(n) for n in element.normalOrder))

                for i in range(0, int(float(element.duration.quarterLength)/0.25)):
                    curr_time = [REST]*12

                    for n in element.normalOrder:
                        note_class = n
                        if i == 0:
                            curr_time[note_class] = B_NOTE
                        else:
                            curr_time[note_class] = I_NOTE

                    notes.append(curr_time)
            elif isinstance(element, note.Rest):
                for i in range(0, int(float(element.duration.quarterLength)/0.25)):
                    curr_time = [REST]*12

                    notes.append(curr_time)


    str_notation_notes = []
    for time in notes:
        str_notation_notes.append('.'.join(time))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(str_notation_notes, filepath)

    return str_notation_notes

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

def create_network(network_input, n_vocab, pitchnames):
    """ create the structure of the neural network """
    input_layer = Input(shape=(None,))
    emb_node = Embedding(input_dim=len(pitchnames)+1, output_dim=50)(input_layer)
    x = LSTM(
        256,
        return_sequences=True,
        recurrent_dropout=0.2,
    )(emb_node)
    x = LSTM(
        256,
        return_sequences=True,
        recurrent_dropout=0.2,
    )(x)
    x = LSTM(
        256,
    )(x)
    x = Dense(512)(x)
    output_layer = Dense(n_vocab, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, network_input, network_output):
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

    train_seq = TrainSequence(network_input, network_output, 128)

    model.fit_generator(
        generator=train_seq,
        epochs=100,
        verbose=1,
        shuffle=False,
        callbacks=callbacks_list
    )



class TrainSequence(Sequence):

    def __init__(self, x, y, batch_size=1, preprocess=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        return numpy.asarray(batch_x), numpy.asarray(batch_y)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)



if __name__ == '__main__':
    train_network()
