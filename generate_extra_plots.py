import pandas as pd
import matplotlib.pyplot as plt
import os

# Pastikan folder models ada
os.makedirs('models', exist_ok=True)

# 1. Buka file hasil evaluasi (dari step5_evaluation.py)
df = pd.read_csv('models/test_evaluation_results.csv')

# 2. Scatter plot RUL Aktual vs Prediksi (dengan garis ideal)
plt.figure(figsize=(8, 6))
plt.scatter(df['True_RUL'], df['Pred_RUL'], alpha=0.6, edgecolors='k', s=50)
plt.plot([0, 130], [0, 130], 'r--', linewidth=2, label='Ideal (y=x)')
plt.xlabel('RUL Aktual (Cycles)')
plt.ylabel('RUL Prediksi (Cycles)')
plt.title('Perbandingan RUL Aktual vs Prediksi (100 Unit Mesin)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('models/scatter_rul_actual_vs_pred.png', dpi=300)
plt.close()

print("✅ Scatter plot disimpan: models/scatter_rul_actual_vs_pred.png")

# 3. Histogram error (Prediksi - Aktual)
errors = df['Pred_RUL'] - df['True_RUL']
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=20, edgecolor='black', color='skyblue')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Error Nol')
plt.xlabel('Error (Prediksi - Aktual) [Cycles]')
plt.ylabel('Frekuensi')
plt.title('Distribusi Error Prediksi RUL')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('models/histogram_error.png', dpi=300)
plt.close()

print("✅ Histogram error disimpan: models/histogram_error.png")

# 4. Boxplot error per status kesehatan
def get_status(rul):
    if rul < 25:
        return 'Critical'
    elif rul < 50:
        return 'Caution'
    else:
        return 'Healthy'

df['Status'] = df['True_RUL'].apply(get_status)
df['Error'] = df['Pred_RUL'] - df['True_RUL']

plt.figure(figsize=(8, 6))
df.boxplot(column='Error', by='Status', grid=True)
plt.title('Boxplot Error Prediksi Berdasarkan Status Mesin')
plt.suptitle('')  # hilangkan judul otomatis dari pandas
plt.xlabel('Status Mesin')
plt.ylabel('Error (Prediksi - Aktual) [Cycles]')
plt.tight_layout()
plt.savefig('models/boxplot_error_by_status.png', dpi=300)
plt.close()

print("✅ Boxplot error disimpan: models/boxplot_error_by_status.png")
print("Semua gambar selesai dibuat. Silakan sisipkan ke laporan.")