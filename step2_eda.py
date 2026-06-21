import pandas as pd
import matplotlib.pyplot as plt

columns = ['id', 'cycle', 'os1', 'os2', 'os3'] + ['s' + str(i) for i in range(1, 22)]
train = pd.read_csv('Datasets/CMaps/train_FD001.txt', sep='\s+', header=None, names=columns)

stats = train.describe().T
constant_sensors = stats[stats['std'] == 0].index.tolist()
print("Nilai unik s6:", train['s6'].unique())
print(f"Sensor yang akan dihapus (konstan): {constant_sensors}")

sensors_to_plot = ['s2', 's3', 's4', 's7', 's11', 's12', 's15', 's17', 's20', 's21']

engine_1 = train[train['id'] == 1]

plt.figure(figsize=(15, 12))
for i, sensor in enumerate(sensors_to_plot, 1):
    plt.subplot(5, 2, i)
    plt.plot(engine_1['cycle'], engine_1[sensor], color='tab:red', linewidth=1.5)
    plt.title(f'Tren Sensor {sensor} (Unit 1)', fontsize=10)
    plt.xlabel('Cycle')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualisasi_sensor_eda.png', dpi=300)
plt.show()

print("EDA Selesai. Gambar visualisasi telah disimpan.")