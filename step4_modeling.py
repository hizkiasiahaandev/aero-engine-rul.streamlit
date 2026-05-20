import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

if not os.path.exists('models'):
    os.makedirs('models')

train = pd.read_csv('Datasets/CMaps/train_preprocessed.csv')

def create_sequences(data, seq_length, features):
    x, y = [], []
    for unit_id in data['id'].unique():
        unit_data = data[data['id'] == unit_id]
        feature_data = unit_data[features].values
        target_data = unit_data['RUL'].values
        
        for i in range(len(unit_data) - seq_length):
            x.append(feature_data[i:i+seq_length])
            y.append(target_data[i+seq_length])
    return np.array(x), np.array(y)

features = ['os1', 'os2', 's2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
seq_length = 50

X_train, y_train = create_sequences(train, seq_length, features)

model = Sequential([
    LSTM(units=100, return_sequences=True, input_shape=(seq_length, len(features))),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(
    optimizer='adam', 
    loss='mean_squared_error', 
    metrics=['mae']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=64, 
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

history_df = pd.DataFrame(history.history)
history_df.to_csv('models/training_history.csv', index=False)

model.save('models/lstm_model.keras')

print("Proses Selesai:")
print("- Model asli disimpan: models/lstm_model.keras")
print("- History asli disimpan: models/training_history.csv")