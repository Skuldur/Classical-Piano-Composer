from pydub import AudioSegment
import glob
from music21 import converter, instrument, note, chord
import json
import pickle
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

notes = []

for file in glob.glob("midi_songs/*.mid"):
  print(file)

  midi = converter.parse(file)

  try:
    s2 = instrument.partitionByInstrument(midi)

    for element in s2.parts[0].recurse():
      if isinstance(element, note.Note):
        notes.append(str(element.pitch))
      elif isinstance(element, chord.Chord):
        notes.append('.'.join(str(n) for n in element.normalOrder))
  except:
    print('Midi file does not contain instrumental partition. Skipping...')

    
# get all pitch names
pitchnames = sorted(set(item for item in notes))

with open('data/notes', 'wb') as fp:
    pickle.dump(notes, fp)

# create a dictionary to map pitches to integers
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

# we want to store this to save work for the predictions
with open('data/note_to_int.json', 'w') as file:
     file.write(json.dumps(note_to_int))

# get amount of pitch names
n_vocab = len(set(notes))

sequence_length = 100

input = []
output = []

# create input sequences and the corresponding outputs
for i in range(0, len(notes) - sequence_length, 1):
  sequence_in = notes[i:i + sequence_length]
  sequence_out = notes[i + sequence_length]
  input.append([note_to_int[char] for char in sequence_in])
  output.append(note_to_int[sequence_out])

n_patterns = len(input)

input = numpy.reshape(input, (n_patterns, sequence_length, 1))

input = input / float(n_vocab)

output = np_utils.to_categorical(output)

model = Sequential()
model.add(LSTM(512, input_shape=(input.shape[1], input.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# fit the model
model.fit(input, output, epochs=200, batch_size=64, callbacks=callbacks_list)