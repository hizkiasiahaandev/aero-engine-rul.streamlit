import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import load
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="RUL Dashboard", layout="wide")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('models/lstm_model.keras')
    scaler = load('models/scaler.pkl')
    return model, scaler

model, scaler = load_assets()

with st.sidebar:
    st.title("⚙️ Control Panel")
    uploaded_file = st.file_uploader("Upload Test Dataset", type=['txt', 'csv'])
    st.divider()
    
    if uploaded_file is not None:
        columns = ['id', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
        features = ['os1', 'os2', 's2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
        test_df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, names=columns)
        
        selected_unit = st.sidebar.selectbox("Pilih ID Unit Mesin", test_df['id'].unique())
        sensor_to_plot = st.sidebar.selectbox("Pilih Sensor Visualisasi", [f for f in features if f.startswith('s')])
        
        st.info("Model: LSTM (NASA FD001)")

st.title("🚀 Aero-Engine RUL Predictor")

if uploaded_file is None:
    st.info("Silakan unggah file dataset (test_FD001.txt) di sidebar untuk memulai analisis.")
else:
    unit_data = test_df[test_df['id'] == selected_unit].copy()
    raw_sensor_data = unit_data.copy()
    unit_data[features] = scaler.transform(unit_data[features])
    
    if len(unit_data) < 50:
        padding = np.zeros((50 - len(unit_data), len(features)))
        input_seq = np.vstack((padding, unit_data[features].values))
        st.warning(f"Unit {selected_unit} menggunakan padding (Data < 50 cycle).")
    else:
        input_seq = unit_data[features].values[-50:]
    
    input_seq = input_seq.reshape(1, 50, len(features))
    prediction = model.predict(input_seq, verbose=0)
    
    rul_val = int(np.clip(prediction[0][0], 0, 125))

    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        st.metric("Estimasi RUL", f"{rul_val} Cycles")
        st.write(f"Total Cycle Saat Ini: **{len(unit_data)}**")

    with col2:
        st.write("### Status Mesin")
        if rul_val < 25:
            st.error("🚨 CRITICAL (Segera Maintenance)")
        elif rul_val < 60:
            st.warning("⚠️ CAUTION (Jadwalkan Cek)")
        else:
            st.success("✅ HEALTHY (Kondisi Baik)")
            
    with col3:
        st.subheader(f"Tren {sensor_to_plot}")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(raw_sensor_data['cycle'], raw_sensor_data[sensor_to_plot], color='#FF4B4B')
        ax.set_xlabel('Cycle')
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)

    st.divider()
    
  # Ubah baris ini:
    tab1, tab2, tab3 = st.tabs(["📊 Data Normalized", "📈 Training History (Real)", "📜 Laporan Performa"])
    
    with tab1:
        st.subheader("Data Sensor Terakhir (Normalized)")
        st.dataframe(unit_data.tail(5), width='stretch')
        
    with tab2:
        if os.path.exists('models/training_history.csv'):
            history_df = pd.read_csv('models/training_history.csv')
            c1, c2 = st.columns(2)
            
            with c1:
                st.write("#### Model Loss (MSE)")
                fig_loss, ax_loss = plt.subplots(figsize=(5, 4))
                ax_loss.plot(history_df['loss'], label='Train Loss')
                ax_loss.plot(history_df['val_loss'], label='Val Loss')
                ax_loss.set_xlabel("Epoch")
                ax_loss.legend()
                plt.tight_layout()
                st.pyplot(fig_loss)
                
            with c2:
                st.write("#### Model Accuracy (MAE)")
                fig_mae, ax_mae = plt.subplots(figsize=(5, 4))
                ax_mae.plot(history_df['mae'], label='Train MAE', color='orange')
                ax_mae.plot(history_df['val_mae'], label='Val MAE', color='green')
                ax_mae.set_xlabel("Epoch")
                ax_mae.legend()
                plt.tight_layout()
                st.pyplot(fig_mae)
        else:
            st.error("File history training tidak ditemukan. Jalankan step4_modeling.py terlebih dahulu.")
        
        with tab3:
            st.markdown(f"""
    ### Statistik Evaluasi Akhir (Data Test NASA FD001)
    * **Mean Absolute Error (MAE):** 10.64
    * **Root Mean Squared Error (RMSE):** 14.78
    * **Input Sequence:** 50 Cycles
    * **Normalization:** MinMaxScaler (0-1)
    * **Dataset:** NASA C-MAPSS Sub-dataset FD001
    """)
    st.info("Keterangan: Angka ini didasarkan pada pengujian terhadap 100 unit mesin yang belum pernah dilihat model sebelumnya.")