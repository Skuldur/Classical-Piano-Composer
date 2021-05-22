from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Activation


def create_network(network_input, n_vocab, weights_filename=None):
    """Create the structure of the neural network"""
    model = Sequential()
    model.add(
        LSTM(
            512,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            return_sequences=True,
        )
    )
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

    if weights_filename:
        print(f"Loading weights from {weights_filename}")
        model.load_weights(weights_filename)

    return model
