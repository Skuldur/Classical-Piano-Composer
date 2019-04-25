""" This module generates notes for a midi file using the
    trained neural network """
import pickle
import numpy
from music21 import instrument, note, stream, chord, duration
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
import keras.layers as layers
from keras.layers import Dense, CuDNNLSTM, LSTM, concatenate, SpatialDropout1D, Bidirectional, Embedding, Input, Dropout, TimeDistributed, GlobalAveragePooling1D
from keras.models import Model, model_from_json

def generate():
    """ Generate a piano midi file """
    #load the notes used to train the model
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    with open('data/durations', 'rb') as filepath:
        durations = pickle.load(filepath)        

    # Get all pitch names
    pitchnames = sorted(set(notes))
    unique_durations = sorted(set(durations))
    # Get all pitch names
    n_vocab = len(set(notes))
    n_durations = len(set(notes))
    rest_note = '.'.join(['0']*12)
    new_notes = []
    for idx, note in enumerate(notes):
        if note != rest_note:
            new_notes.append(note)

    notes = new_notes

    network_input = prepare_sequences(notes, pitchnames, n_vocab)
    durations_input = prepare_sequences(durations, unique_durations, n_durations)
    model = create_network(n_vocab, pitchnames, unique_durations)
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab, durations_input, unique_durations)
    create_midi(prediction_output)

def prepare_sequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # map between notes and integers and back
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

    # reshape the input into a format compatible with LSTM layers
    #normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    #normalized_input = normalized_input / float(n_vocab)

    #return (network_input, normalized_input)
    return network_input

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

    # Load the weights to each node
    model.load_weights('weights-improvement-233-0.0181-bigger.hdf5')

    return model

def generate_notes(model, network_input, pitchnames, n_vocab, durations_input, unique_durations):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    int_to_duration = dict((number, duration) for number, duration in enumerate(unique_durations))

    notes_pattern = network_input[0]
    durations_pattern = durations_input[0]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        notes_pred_input = numpy.reshape(notes_pattern, (1, len(notes_pattern)))
        durations_pred_input = numpy.reshape(durations_pattern, (1, len(durations_pattern)))
        #prediction_input = prediction_input / float(n_vocab)
        #print(prediction_input)
        notes_prediction, duration_prediction  = model.predict([notes_pred_input, durations_pred_input], verbose=0)

        #print(notes_prediction)

        #notes_index = sample(notes_prediction[0], 0.3)
        #duration_index = sample(duration_prediction[0], 0.3)
        notes_index = numpy.argmax(notes_prediction[0])
        duration_index = numpy.argmax(duration_prediction[0])
        print('the index', notes_index)
        note_result = int_to_note[notes_index]
        duration_result = int_to_duration[duration_index]
        prediction_output.append([note_result, duration_result])

        notes_pattern.append(notes_index)
        notes_pattern = notes_pattern[1:len(notes_pattern)]
        durations_pattern.append(duration_index)
        durations_pattern = durations_pattern[1:len(durations_pattern)]

    return prediction_output

def create_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0.
    output_notes = []

    print(prediction_output)

    # create note and chord objects based on the values generated by the model
    for pattern, pattern_duration in prediction_output:
        if '/' in pattern_duration:
            den, num = pattern_duration.split('/')
            pattern_duration = float(den) / float(num)
        d = duration.Duration()
        d.quarterLength = float(pattern_duration)


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
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern, duration=d)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += float(pattern_duration)

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='test_output.mid')

def sample(preds, temperature=1.0):
    preds = numpy.asarray(preds).astype('float64')
    preds = numpy.log(preds) / temperature
    exp_preds = numpy.exp(preds)
    preds = exp_preds / numpy.sum(exp_preds)
    probas = numpy.random.multinomial(1, preds, 1)
    return numpy.argmax(probas)


if __name__ == '__main__':
    generate()
