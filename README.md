# Classical Piano Composer

This project allows you to train a neural network to generate midi music files that make use of a single instrument

## Requirements

* Python 3.x
* Installing the following packages using pip:
	* Music21
	* Keras
	* Tensorflow
	* h5py

## Training

To train the network you run **musicLSTM.py**. Required to have a midi folder, a folder named `notes_data` to store the generated notes, and 
a folder named `trained_weights` to store the .hdf5 files generated from training. 

E.g.

```
from musicLSTM import train_network
train_network()
```

The network will use every midi file in ./midi_songs to train the network. The midi files should only contain a single instrument to get the most out of the training.

**NOTE**: You can stop the process at any point in time and the weights from the latest completed epoch will be available for text generation purposes.

## Generating music

Once you have trained the network you can generate text using **musicPredict.py**. Select the note files from `notes_data` folder and weights file from `trained_weights`
folder, then save the generated midi output into a folder named `generated_songs`. 

E.g.

```
from musicPredict import generate_midi
generate_midi()
```

You can run the prediction file right away using the **weights.hdf5** file
