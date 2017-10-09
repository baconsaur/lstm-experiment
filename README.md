Time series prediction on historical stock data using an LSTM with Keras/Tensorflow

How to run (using conda):
```bash
conda env create
source activate lstm-experiment
python lstm.py
```

The sample datasets are big, so training can take a long time. You may want to set a lower num_epochs to reduce training time, or train it on less historical data.