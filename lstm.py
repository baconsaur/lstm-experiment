import numpy
import pandas
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model


def split_input_output(sequence):
    X, y = [], []
    for i in range(1, len(sequence)):
        X.append(sequence[i - 1])
        y.append(sequence[i])
    return numpy.array(X), numpy.array(y)


# Load data
dataframe = pandas.read_csv('VZ.csv').drop('Date', 1)[-500:] # Only last 500 days for faster training
dataset = dataframe.values
dataset = dataset.astype('float32')

# Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Split into train/test
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
train_X, train_y = split_input_output(train)
test_X, test_y = split_input_output(test)

# Reshape dataset for LSTM
train_X = numpy.reshape(train_X, (len(train_X), 1, train_X.shape[1]))
test_X = numpy.reshape(test_X, (len(test_X), 1, test_X.shape[1]))

# # Load model
# model = load_model('current_model.h5')

# Train model
batch_size = 1
num_epochs = 500
print('Training for {} epochs...'.format(num_epochs))

model = Sequential()
model.add(LSTM(10, batch_input_shape=(batch_size, 1, train_X.shape[1]), stateful=True))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

for i in range(num_epochs):
    print('Epoch {}'.format(i + 1))
    model.fit(train_X, train_y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
    model.reset_states()

# Save model
model.save('current_model.h5')

# Predict
predict_train = scaler.inverse_transform(model.predict(train_X, batch_size=batch_size))
predict_test = scaler.inverse_transform(model.predict(test_X, batch_size=batch_size))

# Plot
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[0:len(predict_train), :] = predict_train

testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(predict_train):len(dataset) - 2, :] = predict_test

plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
