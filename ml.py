from keras.models import Sequential
from keras.layers import Dense


def ann(dimension: 'int > 0'):
    """Artificial Neural Network algorithm"""
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=dimension))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def cnn(shapes: tuple):
    """Convolutional Neural Network algorithm"""
    from keras.layers import Conv1D, MaxPooling1D, Flatten
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=shapes))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def lstm(shapes: tuple):
    """Long Short-Term Memory algorithm"""
    from keras.layers import LSTM, Dropout
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=shapes))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
