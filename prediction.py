"""Predict stock price at 'Close'"""
import os

import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt

import util
import ml

PLOT_DIR = "plots/ericsson/5y"
INPUT_FILENAME = os.path.join("resources", "ERIC.5y.csv")
COMPANY_NAME = "Ericsson"
UNIT = "USD"

LAGS = 10
NUM_FEATURES = 1  # univariate time-series
EPOCH = 1000
show = False

scaler = StandardScaler()
callback_early_stopping = EarlyStopping(monitor='val_loss', patience=10)

df = pd.read_csv(INPUT_FILENAME)
r, c = df.shape
training_sample_number = r // 10 * 8
validation_sample_number = r // 10
test_sample_number = r - training_sample_number - validation_sample_number

# "Close" stock price at the 5th column
training_df = df.iloc[:training_sample_number, 4:5]
validation_df = df.iloc[training_sample_number:training_sample_number + validation_sample_number, 4:5]
test_df = df.iloc[training_sample_number + validation_sample_number:, 4:5]

# Scale/standardise/normalise the data before model fitting
training_set_scaled = scaler.fit_transform(np.array(training_df))

# Training/validation datasets
X_training, y_training = util.split_dataset_into_X_y(training_set_scaled, LAGS)
X_training_ann = np.reshape(X_training, (-1, X_training.shape[1]))
X_training_deep = np.reshape(X_training, (X_training.shape[0], X_training.shape[1], NUM_FEATURES))

validation_set_scaled = scaler.transform(np.array(validation_df))
X_validation, y_validation = util.split_dataset_into_X_y(validation_set_scaled, LAGS)
X_validation_ann = np.reshape(X_validation, (-1, X_validation.shape[1]))
X_validation_deep = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], NUM_FEATURES))

X_training_svm = np.concatenate((X_training, X_validation))
y_training_svm = np.concatenate((y_training, y_validation))

# Test datasets
test_set_scaled = scaler.transform(np.array(test_df))
X_test, _ = util.split_dataset_into_X_y(test_set_scaled, LAGS)
X_test_svm = X_test
X_test_ann = np.reshape(X_test, (-1, X_test.shape[1]))
X_test_deep = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], NUM_FEATURES))

actual_y = np.array(test_df[LAGS:])

# SVM
SVR_model = SVR().fit(X_training_svm, y_training_svm)

predicted_y_svm = SVR_model.predict(X_test_svm)
predicted_y_svm = scaler.inverse_transform(predicted_y_svm)
print('RMSE SVM: {}'.format(util.rmse(actual_y, predicted_y_svm)))
print('MAPE SVM: {} (%)'.format(util.mape(actual_y, predicted_y_svm)))

predicted_y_svm = np.append(np.repeat(np.nan, LAGS), predicted_y_svm)
predicted_y_svm = pd.DataFrame(predicted_y_svm, columns=['Close'])
predicted_y_svm.index = test_df.index

# ANN
ANN_model = ml.ann(X_training_ann.shape[1])
ANN_model.fit(X_training_ann, y_training, validation_data=(X_validation_ann, y_validation), epochs=EPOCH,
              callbacks=[callback_early_stopping])

predicted_y_ann = ANN_model.predict(X_test_ann, verbose=0)
predicted_y_ann = scaler.inverse_transform(predicted_y_ann)
print('RMSE ANN: {}'.format(util.rmse(actual_y, predicted_y_ann)))
print('MAPE ANN: {} (%)'.format(util.mape(actual_y, predicted_y_ann)))
predicted_y_ann = np.append(np.repeat(np.nan, LAGS), predicted_y_ann)
predicted_y_ann = pd.DataFrame(predicted_y_ann, columns=['Close'])
predicted_y_ann.index = test_df.index

# CNN
CNN_model = ml.cnn((X_training_deep.shape[1], NUM_FEATURES))

CNN_model.fit(X_training_deep, y_training, validation_data=(X_validation_deep, y_validation), epochs=EPOCH,
              callbacks=[callback_early_stopping])

predicted_y_cnn = CNN_model.predict(X_test_deep, verbose=0)
predicted_y_cnn = scaler.inverse_transform(predicted_y_cnn)
print('RMSE CNN: {}'.format(util.rmse(actual_y, predicted_y_cnn)))
print('MAPE CNN: {} (%)'.format(util.mape(actual_y, predicted_y_cnn)))
predicted_y_cnn = np.append(np.repeat(np.nan, LAGS), predicted_y_cnn)
predicted_y_cnn = pd.DataFrame(predicted_y_cnn, columns=['Close'])
predicted_y_cnn.index = test_df.index

# LSTM
LSTM_model = ml.lstm((X_training_deep.shape[1], NUM_FEATURES))

LSTM_model.fit(X_training_deep, y_training, validation_data=(X_validation_deep, y_validation), epochs=EPOCH,
               callbacks=[callback_early_stopping])

predicted_y_lstm = LSTM_model.predict(X_test_deep)
predicted_y_lstm = scaler.inverse_transform(predicted_y_lstm)
print('RMSE LSTM: {}'.format(util.rmse(actual_y, predicted_y_lstm)))
print('MAPE LSTM: {} (%)'.format(util.mape(actual_y, predicted_y_lstm)))
predicted_y_lstm = np.append(np.repeat(np.nan, LAGS), predicted_y_lstm)
predicted_y_lstm = pd.DataFrame(predicted_y_lstm, columns=['Close'])
predicted_y_lstm.index = test_df.index

# Plot
util.mkdir_if_not_exists(PLOT_DIR)
plot_filename = os.path.join(PLOT_DIR, COMPANY_NAME + "_{}.png")
index_df = df.loc[training_sample_number + validation_sample_number:, "Date"]
interval = index_df.shape[0] // 5

for predicted, alg in zip([predicted_y_svm, predicted_y_ann, predicted_y_cnn, predicted_y_lstm],
                          ["SVM", "ANN", "CNN", "LSTM"]):
    fig = plt.figure(figsize=[8, 3])
    ax = fig.add_subplot(111)
    plt.plot(index_df, test_df, color="r", label="Real {} Stock Price".format(COMPANY_NAME))
    plt.plot(index_df, predicted, color="b", label="Predicted {} Stock Price by {}".format(COMPANY_NAME, alg))
    plt.xticks(np.arange(0, test_sample_number, interval))
    plt.title("{} Stock Price Prediction".format(COMPANY_NAME))
    plt.ylabel("{} Stock Price ({})".format(COMPANY_NAME, UNIT))
    plt.legend()
    if show:
        plt.show()
    plt.savefig(plot_filename.format(alg))

fig = plt.figure(figsize=[14, 10])
ax = fig.add_subplot(111)
plt.plot(index_df, test_df, color='k', linewidth=2, label="Real Stock Price".format(COMPANY_NAME))
plt.plot(index_df, predicted_y_svm, color='g', marker='.', linewidth=0.7, label="Predicted by {}".format("SVM"))
plt.plot(index_df, predicted_y_ann, color='y', marker='^', linewidth=0.7, label="Predicted by {}".format("ANN"))
plt.plot(index_df, predicted_y_cnn, color='b', marker='+', linewidth=0.7, label="Predicted by {}".format("CNN"))
plt.plot(index_df, predicted_y_lstm, color='r', marker='x', linewidth=0.7, label="Predicted by {}".format("LSTM"))
plt.xticks(np.arange(0, test_sample_number, interval))
plt.title("{} Stock Price Prediction".format(COMPANY_NAME), fontsize=20)
plt.ylabel("{} Stock Price ({})".format(COMPANY_NAME, UNIT), fontsize=15)
plt.legend(loc='best', fontsize=10)
if show:
    plt.show()
plt.savefig(plot_filename.format("all"))

plt.close('all')
