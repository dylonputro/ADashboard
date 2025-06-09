import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import prepro
import ollama  # jika tidak dipakai, bisa dihapus
import joblib
from darts.models import NBEATSModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries, holidays_timeseries
from darts.metrics import mape

st.set_page_config(layout="wide", page_title="Dashboard Group 15", page_icon="📊")
st.title("📊 Dashboard By Group 15")

# Fungsi untuk membaca data
def load_data(path: str):
    return pd.read_csv(path)

# Inisialisasi session state
if "page" not in st.session_state:
    st.session_state.page = "upload"
if "df" not in st.session_state:
    st.session_state.df = None
if "column_mapping" not in st.session_state:
    st.session_state.column_mapping = {}

# Upload Page
if st.session_state.page == "upload":
    uppath = st.file_uploader("Drop your file please")
    if uppath is None:
        st.stop()
    st.session_state.df = load_data(uppath)

    if st.session_state.df is not None:
        with st.expander("Data Preview"):
            st.dataframe(st.session_state.df)
            st.write(st.session_state.df.dtypes)

        # Standarisasi nama kolom
        standard_columns = ['Tanggal & Waktu', 'ID Struk', 'Tipe Penjualan', 'Nama Pelanggan', 'Nama Produk', 'Kategori', 'Jumlah Produk', 'Harga Produk', 'Metode Pembayaran']
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
            if st.button("Change"):
                st.session_state.df = prepro.fix_column_name(st.session_state.df, st.session_state.column_mapping)

    if st.button("Continue"):
        st.session_state.df = prepro.clean_data(st.session_state.df)
        st.session_state.page = "Dashboard"
        st.rerun()

# Dashboard Page
elif st.session_state.page == "Dashboard":
    df = st.session_state.df
    salesVsTime = prepro.prep_sales(df)
    groupByCustomer = prepro.prep_customer(df)
    groupByHour = prepro.prep_grouphour(df)
    groupByProduct = prepro.prep_groupProduct(df)
    groupByKategori = prepro.prep_groupKategori(df)

    # Metric Cards
    with st.container():
        col1, col2, col3 = st.columns(3)
        for col, label, value, delta in zip(
            [col1, col2, col3],
            ["Rata-Rata Pemasukan Harian", "Rata-Rata Produk Harian", "Rata-Rata Transaksi Harian"],
            [salesVsTime['nominal_transaksi'].mean(), salesVsTime['banyak_produk'].mean(), salesVsTime['banyak_transaksi'].mean()],
            [
                salesVsTime["nominal_transaksi"].iloc[-1] - salesVsTime["nominal_transaksi"].iloc[-2],
                salesVsTime["banyak_produk"].iloc[-1] - salesVsTime["banyak_produk"].iloc[-2],
                salesVsTime["banyak_transaksi"].iloc[-1] - salesVsTime["banyak_transaksi"].iloc[-2],
            ]
        ):
            with col:
                fig = go.Figure(go.Indicator(
                    mode="number+delta",
                    value=value,
                    delta={"reference": value - delta, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
                    title={"text": label},
                    number={"font": {"size": 60, "color": "#1F2A44"}}
                ))
                fig.update_layout(width=400, height=150)
                st.plotly_chart(fig, use_container_width=True)

    # Time Series Plots
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.line(salesVsTime, x="Tanggal & Waktu", y="banyak_transaksi", title="Banyak Transaksi Seiring Waktu"), use_container_width=True)
        with col2:
            st.plotly_chart(px.line(groupByHour, x="Jam", y="Jumlah_produk", title="Rata-rata Produk per Jam"), use_container_width=True)

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.line(salesVsTime, x="Tanggal & Waktu", y="banyak_jenis_produk", title="Ragam Produk Seiring Waktu"))
        with col2:
            loaded_model = NBEATSModel.load("forecasting_model20.pth")
            loaded_scaler = joblib.load("scaler.save")
            forecast = loaded_model.predict(n=7)
            forecast_actual = loaded_scaler.inverse_transform(forecast)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=salesVsTime['Tanggal & Waktu'], y=salesVsTime['nominal_transaksi'], mode='lines', name='Actual'))
            fig.add_trace(go.Scatter(x=forecast_actual.time_index, y=forecast_actual.univariate_values(), mode='lines', name='Forecast', line=dict(dash='dash')))
            fig.update_layout(title="Forecasting Penjualan 7 Hari", xaxis_title="Tanggal", yaxis_title="Penjualan")
            st.plotly_chart(fig)

    # Produk dan Kategori
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            top_8 = groupByProduct.nlargest(8, "Jumlah_produk")
            other_sum = groupByProduct.loc[~groupByProduct["Nama Produk"].isin(top_8["Nama Produk"]), "Jumlah_produk"].sum()
            other_row = pd.DataFrame([{"Nama Produk": "Lainnya", "Jumlah_produk": other_sum}])
            pie_df = pd.concat([top_8, other_row], ignore_index=True)
            st.plotly_chart(px.pie(pie_df, names="Nama Produk", values="Jumlah_produk", hole=0.4, title="Produk Terlaris"))
        with col2:
            st.plotly_chart(px.line(df, x="Tanggal & Waktu", y="Jumlah Produk", color="Kategori", title="Produk per Hari per Kategori"))

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.bar(groupByKategori, x="Kategori", y="Total_omset", color="Kategori", title="Total Omset per Kategori"))
        with col2:
            st.plotly_chart(px.scatter(df, x="Jumlah Produk", y="Harga Produk", color="Kategori", title="Harga vs Jumlah Produk", symbol="Kategori"))

    # Segmentasi Customer
    groupByCustomer = prepro.customer_segmentation(groupByCustomer)
    with st.container():
        cluster_counts = groupByCustomer["cluster"].value_counts().reset_index()
        cluster_counts.columns = ["cluster", "count"]
        st.plotly_chart(px.bar(cluster_counts, x="cluster", y="count", color="cluster", title="Distribusi Klaster Pelanggan"))

        option = st.selectbox("Pilih Cluster", ["All"] + groupByCustomer["cluster"].unique().tolist())
        df_filtered = groupByCustomer if option == "All" else groupByCustomer[groupByCustomer["cluster"] == option].copy()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(go.Figure(go.Indicator(mode="number", value=df_filtered["totSpen"].mean(), title={"text": "Rata-rata Pengeluaran"})))
        with col2:
            st.plotly_chart(go.Figure(go.Indicator(mode="number", value=df_filtered["totJum"].mean(), title={"text": "Rata-rata Produk"})))
        with col3:
            st.plotly_chart(go.Figure(go.Indicator(mode="number", value=df_filtered["frequency"].mean(), title={"text": "Rata-rata Frekuensi"})))
