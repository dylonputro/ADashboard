import pandas as pd 
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import prepro
import ollama
import joblib 
from darts.models import NBEATSModel

st.set_page_config(layout="wide", page_title="Dashboard Group 15", page_icon="📊")
st.title("Adashboard By Group 15")

# Fungsi read data 
def load_data(path: str):
    data = pd.read_csv(path)
    return data 

# Initialize session state variables jika belum ada
if "page" not in st.session_state:
    st.session_state.page = "upload"
if "df" not in st.session_state:
    st.session_state.df = None
if "column_mapping" not in st.session_state:
    st.session_state.column_mapping = {}
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi! How can I help you today?"}]

# Sidebar untuk navigasi halaman
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Data", "Dashboard"])

# Set session_state page sesuai pilihan sidebar
st.session_state.page = "upload" if page == "Upload Data" else "Dashboard"

# Halaman Upload
if st.session_state.page == "upload":
    uppath = st.file_uploader("Drop your file please")
    if uppath is None:
        st.warning("Please upload a file to continue.")
        st.stop()
    st.session_state.df = load_data(uppath)
    if st.session_state.df is not None:
        with st.expander("Data Preview"):
            st.dataframe(st.session_state.df)
            st.write(st.session_state.df.dtypes)
        standard_columns = ['Tanggal & Waktu', 'ID Struk', 'Tipe Penjualan', 'Nama Pelanggan','Nama Produk', 'Kategori', 'Jumlah Produk', 'Harga Produk', 'Metode Pembayaran']
        submitted_columns = st.session_state.df.columns.tolist()
        if set(submitted_columns) == set(standard_columns):
            st.success("✅ Column names match the standard.")
        else:
            st.warning("⚠️ Column names do not match the standard!")
            for i, col in enumerate(standard_columns):
                default_value = st.session_state.column_mapping.get(col, None)
                st.session_state.column_mapping[col] = st.selectbox(
                    f"Select column for '{col}'", submitted_columns,
                    index=submitted_columns.index(default_value) if default_value in submitted_columns else 0,
                    key=f"col_{i}"
                )
            if st.button("Change Column Names"):
                st.session_state.df = prepro.fix_column_name(st.session_state.df, st.session_state.column_mapping)
                st.experimental_rerun()
    if st.button("Continue to Dashboard"):
        st.session_state.df = prepro.clean_data(st.session_state.df)
        st.session_state.page = "Dashboard"
        st.experimental_rerun()

# Halaman Dashboard
elif st.session_state.page == "Dashboard":
    if st.session_state.df is None:
        st.warning("Please upload data first in the Upload Data page.")
        st.stop()

    salesVsTime = prepro.prep_sales(st.session_state.df)
    groupByCustomer = prepro.prep_customer(st.session_state.df)
    groupByHour = prepro.prep_grouphour(st.session_state.df)
    groupByProduct = prepro.prep_groupProduct(st.session_state.df)
    groupByKategori = prepro.prep_groupKategori(st.session_state.df)

    st.header("Sales Dashboard")
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="number+delta",
                value=salesVsTime['nominal_transaksi'].mean(),
                title={"text": "Rata-Rata Pemasukan Harian"},
                delta={"reference": salesVsTime['nominal_transaksi'].mean() - (salesVsTime["nominal_transaksi"].iloc[-1] - salesVsTime["nominal_transaksi"].iloc[-2]), "relative": False,
                       "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
                number={"font": {"size": 60, "color": "#1F2A44"}}
            ))
            fig.update_layout(width=400, height=150)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="number+delta",
                value=salesVsTime['banyak_produk'].mean(),
                title={"text": "Rata-Rata Produk Harian"},
                delta={"reference": salesVsTime['banyak_produk'].mean() - (salesVsTime["banyak_produk"].iloc[-1] - salesVsTime["banyak_produk"].iloc[-2]), "relative": False,
                       "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
                number={"font": {"size": 60, "color": "#1F2A44"}}
            ))
            fig.update_layout(width=400, height=150)
            st.plotly_chart(fig, use_container_width=True)
        with col3:
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="number+delta",
                value=salesVsTime['banyak_transaksi'].mean(),
                title={"text": "Rata-Rata Transaksi Dalam Harian"},
                delta={"reference": salesVsTime['banyak_transaksi'].mean() - (salesVsTime["banyak_transaksi"].iloc[-1] - salesVsTime["banyak_transaksi"].iloc[-2]), "relative": False,
                       "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
                number={"font": {"size": 60, "color": "#1F2A44"}}
            ))
            fig.update_layout(width=400, height=150)
            st.plotly_chart(fig, use_container_width=True)

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(salesVsTime, x="Tanggal & Waktu", y="banyak_transaksi", title="Banyak Transaksi Seiring Waktu")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(groupByHour, x="Jam", y="Jumlah_produk", title="Rata-rata Banyak Produk yang dipesan dalam Seharian")
            st.plotly_chart(fig, use_container_width=True)

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(salesVsTime, x="Tanggal & Waktu", y="banyak_jenis_produk", title="Banyak Ragam Produk Seiring Waktu")
            st.plotly_chart(fig)
        with col2:
            loaded_model = NBEATSModel.load("forecasting_model20.pth")
            loaded_scaler = joblib.load("scaler.save")
            forecast_loaded_scaled = loaded_model.predict(n=7)
            forecast_loaded_actual = loaded_scaler.inverse_transform(forecast_loaded_scaled)
            forecast_loaded_values = forecast_loaded_actual.univariate_values()
            forecast_loaded_dates = forecast_loaded_actual.time_index
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=salesVsTime['Tanggal & Waktu'],
                y=salesVsTime['nominal_transaksi'],
                mode='lines',
                name='actual',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=forecast_loaded_dates,
                y=forecast_loaded_values,
                mode='lines',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title='Prediksi Total Penjualan 7 Hari ke Depan',
                xaxis_title='Tanggal',
                yaxis_title='Total Penjualan'
            )
            st.plotly_chart(fig)

    st.header("Product Dashboard")
    with st.container():
        col21, col22 = st.columns(2)
        with col21:
            top_5 = groupByProduct.nlargest(8, "Jumlah_produk")
            other_total = groupByProduct.loc[~groupByProduct["Nama Produk"].isin(top_5["Nama Produk"]), "Jumlah_produk"].sum()
            other_row = pd.DataFrame([{"Nama Produk": "Other", "Jumlah_produk": other_total}])
            top_5 = pd.concat([top_5, other_row], ignore_index=True)
            fig = px.pie(top_5, names="Nama Produk", values="Jumlah_produk", hole=0.4, title="Donut chart Produk")
            st.plotly_chart(fig)
        with col22:
            fig = px.line(st.session_state.df, x="Tanggal & Waktu", y="Jumlah Produk", color="Kategori", title="Line Chart dengan Banyak Garis Berdasarkan Kategori")
            st.plotly_chart(fig)
        col31, col32 = st.columns(2)
        with col31:
            fig = px.bar(groupByKategori, x="Kategori", y="Total_omset", title="Bar Plot Berdasarkan Kategori", color="Kategori")
            st.plotly_chart(fig)
        with col32:
            fig = px.scatter(st.session_state.df, x="Jumlah Produk", y="Harga Produk", color="Kategori", title="Scatter Plot Berdasarkan Kategori", size_max=10, symbol="Kategori")
            st.plotly_chart(fig)

    st.header("Customer Segmentation Dashboard")
    groupByCustomer = prepro.customer_segmentation(groupByCustomer)
    with st.container():
        valueCCount = groupByCustomer["cluster"].value_counts().reset_index()
        valueCCount.columns = ["cluster", "count"]
        fig = px.bar(valueCCount, x="cluster", y="count", color="cluster", title="Bar Chart Jumlah Produk per Kategori")
        fig.update_layout(width=800, height=400)
        st.plotly_chart(fig)
        optionCluster = ["All"] + groupByCustomer["cluster"].unique().tolist()
        option1 = st.selectbox("What cluster ?", optionCluster)
        if option1 == "All":
            clusteringmask = groupByCustomer.copy()
        else:
            clusteringmask = groupByCustomer[groupByCustomer["cluster"] == option1]
        st.dataframe(clusteringmask)

# Jika butuh debug session state
# st.write(st.session_state)
