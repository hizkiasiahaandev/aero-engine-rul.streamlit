import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

columns = ['id', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
train = pd.read_csv('Datasets/CMaps/train_FD001.txt', sep=r'\s+', header=None, names=columns)

def add_rul(df):
    max_cycle = df.groupby('id')['cycle'].transform('max')
    df['RUL'] = max_cycle - df['cycle']
    df['RUL'] = df['RUL'].clip(upper=125)
    return df

train = add_rul(train)

drop_sensors = ['s1', 's5', 's10', 's16', 's18', 's19', 'os3']
train = train.drop(columns=drop_sensors)

features = [col for col in train.columns if col not in ['id', 'cycle', 'RUL']]

scaler = MinMaxScaler()
train[features] = scaler.fit_transform(train[features])

dump(scaler, 'models/scaler.pkl')
train.to_csv('Datasets/CMaps/train_preprocessed.csv', index=False)

print("Preprocessing Selesai:")
print(f"- Sensor konstan dihapus: {drop_sensors}")
print(f"- Scaler disimpan ke: models/scaler.pkl")
print("- Dataset siap training: train_preprocessed.csv")