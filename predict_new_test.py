""" This module generates notes for a midi file using the
    trained neural network """
import pickle
import numpy
from music21 import instrument, note, stream, chord, duration, tempo
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
import keras.layers as layers
from keras.layers import Dense, CuDNNLSTM, Embedding, Input, Dropout, LSTM, concatenate, SpatialDropout1D, Reshape, Activation, Permute, Multiply, Lambda, RepeatVector
from keras.models import Model, model_from_json
from keras import backend as K


def get_set_of_values(matrix):
    s = set()

    for row in matrix:
        for item in row:
            s.add(item)

    return s

def generate():
    """ Generate a piano midi file """
    #load the notes used to train the model
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    with open('data/durations', 'rb') as filepath:
        durations = pickle.load(filepath)

    with open('data/offsets', 'rb') as filepath:
        offsets = pickle.load(filepath) 
    with open('data/time', 'rb') as filepath:
        time = pickle.load(filepath)               

    # Get all pitch names
    pitchnames = sorted(get_set_of_values(notes))
    unique_durations = sorted(get_set_of_values(durations))
    print(unique_durations)
    unique_offsets = sorted(get_set_of_values(offsets))
    unique_time = sorted(get_set_of_values(time))
    # Get all pitch names
    n_vocab = len(pitchnames)
    n_durations = len(unique_durations)
    n_offsets = len(unique_offsets)
    n_time = len(unique_time)

    network_input = prepare_sequences(notes, pitchnames, n_vocab)
    durations_input = prepare_sequences(durations, unique_durations, n_durations)
    offsets_input = prepare_sequences(offsets, unique_offsets, n_offsets)
    time_input = prepare_sequences(time, unique_time, n_time)
    model = create_network(n_vocab, pitchnames, unique_durations, unique_offsets, unique_time)
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab, durations_input, unique_durations, offsets_input, unique_offsets, time_input, unique_time)
    create_midi(prediction_output)

def prepare_sequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 32
    network_input = []
    output = []

    for i, file in enumerate(notes):
        print('File %s starts at %s' % (i, len(network_input)))
        slow = 0
        fast = sequence_length

        while fast < len(file):
            if fast - slow < sequence_length:
                sequence_in = file[slow:fast]
                network_input.append([note_to_int[char] for char in sequence_in])
            elif fast - slow == sequence_length:
                sequence_in = file[slow:fast]
                network_input.append([note_to_int[char] for char in sequence_in])
                slow += 1
            fast += 1

    return network_input

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
    notes_out = Dense(note_output-1, activation='softmax', name='notes')(c)
    duration_out = Dense(len(unique_durations), activation='softmax', name='duration')(c)
    offset_out = Dense(len(unique_offsets), activation='softmax', name='offset')(c)
    time_out = Dense(124, activation='softmax', name='time')(c)


    model = Model(inputs=[notes_in, duration_in, offsets_in, time_in], outputs=[notes_out, duration_out, offset_out, time_out])
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[1, 0.5, 0.5, 0.5], optimizer='rmsprop')

    # Load the weights to each node
    model.load_weights('weights-improvement-381-0.0192-bigger.hdf5')

    return model

def generate_notes(model, network_input, pitchnames, n_vocab, durations_input, unique_durations, offset_input, unique_offset, time_input, unique_time):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(network_input)-1)
    #start = 3191

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    int_to_duration = dict((number, duration) for number, duration in enumerate(unique_durations))
    int_to_offset = dict((number, offset) for number, offset in enumerate(unique_offset))
    int_to_time = dict((number, time) for number, time in enumerate(unique_time))

    notes_pattern = network_input[start]
    durations_pattern = durations_input[start]
    offset_pattern = offset_input[start]
    #time_pattern = time_input[start]

    max_time = numpy.random.randint(1, 5)

    time_pattern = []

    time = 127
    i = 0

    while i < 32:
        for j in range(max_time):
            time_pattern.append(time)
            i += 1

        time -= 1

    print(time_pattern)

    prediction_output = []

    for i in range(0, len(notes_pattern)):
        prediction_output.append([int_to_note[notes_pattern[i]], int_to_duration[durations_pattern[i]], int_to_offset[offset_pattern[i]]])

    #notes_pattern = [0]* 90 + notes_pattern[0:10]
    #durations_pattern = [0]* 90 + durations_pattern[0:10]
    #offset_pattern = [0]* 90 + offset_pattern[0:10]

    # generate 500 notes
    #for note_index in range(1000):
    for i in range(1000):
        notes_pred_input = numpy.reshape(notes_pattern, (1, len(notes_pattern)))
        durations_pred_input = numpy.reshape(durations_pattern, (1, len(durations_pattern)))
        offset_pred_input = numpy.reshape(offset_pattern, (1, len(offset_pattern)))
        time_pred_input = numpy.reshape(time_pattern, (1, len(time_pattern)))
        #prediction_input = prediction_input / float(n_vocab)
        #print(prediction_input)
        notes_prediction, duration_prediction, offset_prediction, time_prediction = model.predict([notes_pred_input, durations_pred_input, offset_pred_input, time_pred_input], verbose=0)

        #print(notes_prediction)

        #notes_index = sample(notes_prediction[0], 0.85)
        #duration_index = sample(duration_prediction[0], 0.2)
        #offset_index = sample(offset_prediction[0], 0.3)
        notes_index = numpy.argmax(notes_prediction)
        duration_index = numpy.argmax(duration_prediction)
        offset_index = numpy.argmax(offset_prediction)
        time_index = numpy.argmax(time_prediction)
        note_result = int_to_note[notes_index]
        duration_result = int_to_duration[duration_index]
        offset_result = int_to_offset[offset_index]
        time_result = int_to_time[time_index]

        print('THE TIME ', time_result)

        prediction_output.append([note_result, duration_result, offset_result])

        if note_result == 'END':
            print(notes_index)
            break

        notes_pattern.append(notes_index)
        notes_pattern = notes_pattern[1:len(notes_pattern)]
        durations_pattern.append(duration_index)
        durations_pattern = durations_pattern[1:len(durations_pattern)]
        offset_pattern.append(offset_index)
        offset_pattern = offset_pattern[1:len(offset_pattern)]
        time_pattern.append(time_index)
        time_pattern = time_pattern[1:len(time_pattern)]

    print(len(prediction_output))

    return prediction_output

def create_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0.
    output_notes = []

    print(prediction_output)

    midi_stream = stream.Stream()

    # create note and chord objects based on the values generated by the model
    for idx, val in enumerate(prediction_output[1:]):
        pattern, pattern_duration, pattern_offset = val

        if pattern == 'END' and pattern_duration == 'inf' and pattern_offset == 'inf':
            print(idx)
            break
        elif pattern == 'END' or pattern_duration == 'inf' or pattern_offset == 'inf':
            print('lolwut')
            break
        d = duration.Duration()
        if pattern_duration == float('inf'):
            pattern_duration = 1.0
        d.quarterLength = float(pattern_duration)

        if idx != 0:
            if pattern_offset == '0.0':
                print("ZERO")
            offset += float(pattern_offset)

        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes, duration=d)
            new_chord.offset = offset
            midi_stream.insert(new_chord)
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern, duration=d)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            midi_stream.insert(new_note)
            output_notes.append(new_note)

    #midi_stream.makeVoices()
    #print(midi_stream.show('text'))

    #midi_stream = stream.Stream(output_notes)
    #print(dir(midi_stream))
    #midi_stream = midi_stream.chordify()
    print('THIS STREAM IS A SEQUENCE (NO OVERLAPS)', midi_stream.isSequence())

    midi_stream.write('midi', fp='test_output.midi')

def sample(preds, temperature=1.0):
    preds = numpy.asarray(preds).astype('float64')
    preds = numpy.log(preds) / temperature
    exp_preds = numpy.exp(preds)
    preds = exp_preds / numpy.sum(exp_preds)
    probas = numpy.random.multinomial(1, preds, 1)
    return numpy.argmax(probas)


if __name__ == '__main__':
    generate()
