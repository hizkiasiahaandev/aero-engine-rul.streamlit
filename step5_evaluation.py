import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import matplotlib.pyplot as plt
from joblib import load

columns = ['id', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
features = ['os1', 'os2', 's2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
seq_length = 50

test_df = pd.read_csv('Datasets/CMaps/test_FD001.txt', sep=r'\s+', header=None, names=columns)
true_rul = pd.read_csv('Datasets/CMaps/RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])

scaler = load('models/scaler.pkl')
test_df[features] = scaler.transform(test_df[features])

model = tf.keras.models.load_model('models/lstm_model.keras')

def get_last_sequences(df, seq_length, features):
    last_seq = []
    for unit_id in df['id'].unique():
        unit_data = df[df['id'] == unit_id]
        if len(unit_data) >= seq_length:
            last_seq.append(unit_data[features].values[-seq_length:])
        else:
            padding = np.zeros((seq_length - len(unit_data), len(features)))
            last_seq.append(np.vstack((padding, unit_data[features].values)))
    return np.array(last_seq)

X_test = get_last_sequences(test_df, seq_length, features)
y_pred = model.predict(X_test)

# Penambahan Clipping: Karena pas training RUL dibatasi 125, 
# maka prediksi gak boleh minus atau lebih dari 125 agar logis.
y_pred = np.clip(y_pred, 0, 125)
y_true = true_rul['RUL'].values

rmse = math.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

# Simpan hasil perbandingan buat lampiran atau Dashboard
results_df = pd.DataFrame({'True_RUL': y_true, 'Pred_RUL': y_pred.flatten()})
results_df.to_csv('models/test_evaluation_results.csv', index=False)

plt.figure(figsize=(12, 6))
plt.plot(y_true, label='RUL Asli', color='blue', marker='o', markersize=3, alpha=0.7)
plt.plot(y_pred, label='RUL Prediksi', color='red', linestyle='--', marker='x', markersize=3)
plt.title('Perbandingan RUL Asli vs Prediksi LSTM (Unit Mesin 1-100)')
plt.xlabel('ID Unit Mesin')
plt.ylabel('Remaining Useful Life (RUL)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('models/evaluation_plot.png', dpi=300)
plt.show()

print(f"\n--- HASIL EVALUASI AKHIR ---")
print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")