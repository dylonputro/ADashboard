# File: dashboard_group15.py

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import prepro
import ollama
import joblib
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.timeseries_generation import holidays_timeseries
from darts.metrics import mape

# Streamlit page configuration
st.set_page_config(layout="wide", page_title="Dashboard Group 15", page_icon="📊")
st.title("📊 Dashboard by Group 15")

# Load CSV data
def load_data(path: str):
    return pd.read_csv(path)

# Session state initialization
if "page" not in st.session_state:
    st.session_state.page = "upload"
if "df" not in st.session_state:
    st.session_state.df = None
if "column_mapping" not in st.session_state:
    st.session_state.column_mapping = {}

# Upload Page
if st.session_state.page == "upload":
    st.header("📂 Upload Data")
    uppath = st.file_uploader("Drop your CSV file below")
    if uppath is None:
        st.stop()

    st.session_state.df = load_data(uppath)

    if st.session_state.df is not None:
        with st.expander("📄 Data Preview"):
            st.dataframe(st.session_state.df)
            st.write(st.session_state.df.dtypes)

        standard_columns = ['Tanggal & Waktu', 'ID Struk', 'Tipe Penjualan', 'Nama Pelanggan',
                            'Nama Produk', 'Kategori', 'Jumlah Produk', 'Harga Produk', 'Metode Pembayaran']
        submitted_columns = st.session_state.df.columns.tolist()

        if set(submitted_columns) == set(standard_columns):
            st.success("✅ Column names match the standard.")
        else:
            st.warning("⚠️ Column names do not match the standard!")
            for i, col in enumerate(standard_columns):
                default_value = st.session_state.column_mapping.get(col, None)
                st.session_state.column_mapping[col] = st.selectbox(
                    f"Select column for '{col}'",
                    submitted_columns,
                    index=submitted_columns.index(default_value) if default_value in submitted_columns else 0,
                    key=f"col_{i}"
                )
            if st.button("🔄 Change"):
                st.session_state.df = prepro.fix_column_name(st.session_state.df, st.session_state.column_mapping)

    if st.button("➡️ Continue"):
        st.session_state.page = "Dashboard"
        st.session_state.df = prepro.clean_data(st.session_state.df)
        st.rerun()

# Dashboard Page
elif st.session_state.page == "Dashboard":
    st.header("📈 Sales & Customer Dashboard")

    # Preprocessing
    salesVsTime = prepro.prep_sales(st.session_state.df)
    groupByCustomer = prepro.prep_customer(st.session_state.df)
    groupByHour = prepro.prep_grouphour(st.session_state.df)
    groupByProduct = prepro.prep_groupProduct(st.session_state.df)
    groupByKategori = prepro.prep_groupKategori(st.session_state.df)

    # Indicator Metrics
    with st.container():
        col1, col2, col3 = st.columns(3)
        metrics = [
            (col1, 'nominal_transaksi', "Rata-Rata Pemasukan Harian"),
            (col2, 'banyak_produk', "Rata-Rata Produk Harian"),
            (col3, 'banyak_transaksi', "Rata-Rata Transaksi Harian")
        ]
        for col, field, title in metrics:
            with col:
                mean_val = salesVsTime[field].mean()
                delta_val = mean_val - (salesVsTime[field].iloc[-1] - salesVsTime[field].iloc[-2])
                fig = go.Figure(go.Indicator(
                    mode="number+delta",
                    value=mean_val,
                    delta={"reference": delta_val, "relative": False, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
                    title={"text": title},
                    number={"font": {"size": 50}}
                ))
                st.plotly_chart(fig, use_container_width=True)

    # Time Series & Forecast
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(salesVsTime, x="Tanggal & Waktu", y="banyak_transaksi", title="📊 Transaksi Seiring Waktu")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(groupByHour, x="Jam", y="Jumlah_produk", title="📊 Produk Harian per Jam")
            st.plotly_chart(fig, use_container_width=True)

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(salesVsTime, x="Tanggal & Waktu", y="banyak_jenis_produk", title="📊 Ragam Produk Harian")
            st.plotly_chart(fig)

        with col2:
            loaded_model = NBEATSModel.load("forecasting_model20.pth")
            loaded_scaler = joblib.load("scaler.save")
            forecast_loaded_scaled = loaded_model.predict(n=7)
            forecast_loaded_actual = loaded_scaler.inverse_transform(forecast_loaded_scaled)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=salesVsTime['Tanggal & Waktu'], y=salesVsTime['nominal_transaksi'],
                                     mode='lines', name='Actual', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=forecast_loaded_actual.time_index, y=forecast_loaded_actual.univariate_values(),
                                     mode='lines', name='Forecast', line=dict(color='red', dash='dash')))
            fig.update_layout(title='📈 Prediksi Penjualan 7 Hari ke Depan', xaxis_title='Tanggal', yaxis_title='Total Penjualan')
            st.plotly_chart(fig)

    # Product Dashboard
    with st.container():
        col1, col2 = st.columns(2)
        top_5 = groupByProduct.nlargest(8, "Jumlah_produk")
        other_total = groupByProduct.loc[~groupByProduct["Nama Produk"].isin(top_5["Nama Produk"]), "Jumlah_produk"].sum()
        top_5 = pd.concat([top_5, pd.DataFrame([{"Nama Produk": "Other", "Jumlah_produk": other_total}])], ignore_index=True)
        with col1:
            fig = px.pie(top_5, names="Nama Produk", values="Jumlah_produk", hole=0.4, title="🍩 Produk Terlaris")
            st.plotly_chart(fig)
        with col2:
            fig = px.line(st.session_state.df, x="Tanggal & Waktu", y="Jumlah Produk", color="Kategori", title="📊 Kategori per Waktu")
            st.plotly_chart(fig)

        col3, col4 = st.columns(2)
        with col3:
            fig = px.bar(groupByKategori, x="Kategori", y="Total_omset", color="Kategori", title="📦 Omset per Kategori")
            st.plotly_chart(fig)
        with col4:
            fig = px.scatter(st.session_state.df, x="Jumlah Produk", y="Harga Produk", color="Kategori", title="🎯 Scatter Plot Harga vs Jumlah")
            st.plotly_chart(fig)

    # Customer Segmentation
    groupByCustomer = prepro.customer_segmentation(groupByCustomer)
    with st.container():
        st.subheader("👥 Customer Segmentation")
        cluster_count = groupByCustomer["cluster"].value_counts().reset_index()
        cluster_count.columns = ["cluster", "count"]
        fig = px.bar(cluster_count, x="cluster", y="count", color="cluster", title="📊 Jumlah Customer per Cluster")
        st.plotly_chart(fig)

        selected_cluster = st.selectbox("Pilih Cluster:", ["All"] + sorted(groupByCustomer["cluster"].unique().tolist()))
        if selected_cluster != "All":
            filtered_data = groupByCustomer[groupByCustomer["cluster"] == selected_cluster]
        else:
            filtered_data = groupByCustomer

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Rata-rata Pengeluaran", f"{filtered_data['totSpen'].mean():,.2f}")
        with col2:
            st.metric("📦 Rata-rata Jumlah Produk", f"{filtered_data['totJum'].mean():.2f}")
        with col3:
            st.metric("🧾 Rata-rata Transaksi", f"{filtered_data['count'].mean():.2f}")
        with col4:
            st.metric("🛍️ Rata-rata Produk per Transaksi", f"{(filtered_data['totJum'] / filtered_data['count']).mean():.2f}")
