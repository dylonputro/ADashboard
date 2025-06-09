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
# App Config
st.set_page_config(page_title="Dashboard Group 15", layout="wide", page_icon="📊")

# Session State Setup
if "page" not in st.session_state:
    st.session_state.page = "Upload"
if "df" not in st.session_state:
    st.session_state.df = None
if "column_mapping" not in st.session_state:
    st.session_state.column_mapping = {}

# Sidebar Navigation
with st.sidebar:
    st.title("📁 Navigation")
    st.markdown("Select a section below:")
    page = st.radio("Menu", ["Upload Data", "Dashboard", "Chatbot"])

    if st.session_state.df is not None:
        st.divider()
        st.markdown("### Dataset Info")
        st.write("✅ Rows:", st.session_state.df.shape[0])
        st.write("✅ Columns:", st.session_state.df.shape[1])

    st.session_state.page = page

# Load CSV Data
def load_data(path: str):
    return pd.read_csv(path)

# Upload Page
if st.session_state.page == "Upload Data":
    st.title("📤 Upload Transaction Data")
    st.markdown("Please upload your CSV file below:")
    uppath = st.file_uploader("Drop your file here (.csv)", type="csv")

    if uppath:
        st.session_state.df = load_data(uppath)
        st.success("File uploaded successfully!")

        with st.expander("🔍 Data Preview"):
            st.dataframe(st.session_state.df)
            st.write(st.session_state.df.dtypes)

        # Validate Column Names
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
                    f"Select column for '{col}'", submitted_columns,
                    index=submitted_columns.index(default_value) if default_value in submitted_columns else 0,
                    key=f"col_{i}"
                )
            if st.button("Fix Column Names"):
                st.session_state.df = prepro.fix_column_name(st.session_state.df, st.session_state.column_mapping)

        if st.button("Proceed to Dashboard"):
            st.session_state.df = prepro.clean_data(st.session_state.df)
            st.session_state.page = "Dashboard"
            st.rerun()

# Dashboard Page
elif st.session_state.page == "Dashboard":
    st.title("📊 Sales & Customer Dashboard")
    salesVsTime = prepro.prep_sales(st.session_state.df)
    groupByCustomer = prepro.prep_customer(st.session_state.df)
    groupByHour = prepro.prep_grouphour(st.session_state.df)
    groupByProduct = prepro.prep_groupProduct(st.session_state.df)
    groupByKategori = prepro.prep_groupKategori(st.session_state.df)

    st.subheader("📈 Daily Metrics Overview")
    col1, col2, col3 = st.columns(3)

    def create_indicator(value, title, delta):
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=value,
            delta={"reference": value - delta, "relative": False,
                   "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
            title={"text": title},
            number={"font": {"size": 60, "color": "#1F2A44"}}
        ))
        fig.update_layout(height=150)
        return fig

    col1.plotly_chart(create_indicator(salesVsTime['nominal_transaksi'].mean(),
                                       "Rata-Rata Pemasukan Harian",
                                       salesVsTime["nominal_transaksi"].iloc[-1] - salesVsTime["nominal_transaksi"].iloc[-2]), use_container_width=True)

    col2.plotly_chart(create_indicator(salesVsTime['banyak_produk'].mean(),
                                       "Rata-Rata Produk Harian",
                                       salesVsTime["banyak_produk"].iloc[-1] - salesVsTime["banyak_produk"].iloc[-2]), use_container_width=True)

    col3.plotly_chart(create_indicator(salesVsTime['banyak_transaksi'].mean(),
                                       "Rata-Rata Transaksi Harian",
                                       salesVsTime["banyak_transaksi"].iloc[-1] - salesVsTime["banyak_transaksi"].iloc[-2]), use_container_width=True)

    st.divider()
    st.subheader("🕒 Waktu vs Aktivitas Transaksi")
    col1, col2 = st.columns(2)

    col1.plotly_chart(px.line(salesVsTime, x="Tanggal & Waktu", y="banyak_transaksi", title="Banyak Transaksi Seiring Waktu"), use_container_width=True)
    col2.plotly_chart(px.line(groupByHour, x="Jam", y="Jumlah_produk", title="Jumlah Produk per Jam"), use_container_width=True)

    col3, col4 = st.columns(2)
    col3.plotly_chart(px.line(salesVsTime, x="Tanggal & Waktu", y="banyak_jenis_produk", title="Ragam Produk Seiring Waktu"))

    # Forecasting
    with col4:
        loaded_model = NBEATSModel.load("forecasting_model20.pth")
        loaded_scaler = joblib.load("scaler.save")
        forecast_scaled = loaded_model.predict(n=7)
        forecast_actual = loaded_scaler.inverse_transform(forecast_scaled)
        forecast_dates = forecast_actual.time_index
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=salesVsTime['Tanggal & Waktu'], y=salesVsTime['nominal_transaksi'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_actual.univariate_values(), mode='lines', name='Forecast', line=dict(dash='dash')))
        fig.update_layout(title="🔮 Forecast Penjualan 7 Hari ke Depan")
        st.plotly_chart(fig)

    st.divider()
    st.subheader("🛍️ Produk & Kategori")

    top_5 = groupByProduct.nlargest(8, "Jumlah_produk")
    other_total = groupByProduct.loc[~groupByProduct["Nama Produk"].isin(top_5["Nama Produk"]), "Jumlah_produk"].sum()
    other_row = pd.DataFrame([{"Nama Produk": "Other", "Jumlah_produk": other_total}])
    top_5 = pd.concat([top_5, other_row], ignore_index=True)

    col1, col2 = st.columns(2)
    col1.plotly_chart(px.pie(top_5, names="Nama Produk", values="Jumlah_produk", hole=0.4, title="Produk Terlaris"))
    col2.plotly_chart(px.line(st.session_state.df, x="Tanggal & Waktu", y="Jumlah Produk", color="Kategori", title="Tren Produk per Kategori"))

    col3, col4 = st.columns(2)
    col3.plotly_chart(px.bar(groupByKategori, x="Kategori", y="Total_omset", color="Kategori", title="Omset per Kategori"))
    col4.plotly_chart(px.scatter(st.session_state.df, x="Jumlah Produk", y="Harga Produk", color="Kategori", title="Harga vs Jumlah Produk"))

    st.divider()
    st.subheader("👥 Customer Segmentation")
    groupByCustomer = prepro.customer_segmentation(groupByCustomer)
    valueCCount = groupByCustomer["cluster"].value_counts().reset_index()
    valueCCount.columns = ["cluster", "count"]

    st.plotly_chart(px.bar(valueCCount, x="cluster", y="count", color="cluster", title="Distribusi Cluster Customer"))

    optionCluster = ["All"] + groupByCustomer["cluster"].unique().tolist()
    selected = st.selectbox("Select Cluster:", optionCluster)
    if selected != "All":
        groupByCustomer = groupByCustomer[groupByCustomer["cluster"] == selected]

    cols = st.columns(4)
    metrics = [("totSpen", "Pengeluaran Rata-rata"),
               ("totJum", "Jumlah Produk Rata-rata"),
               ("totJenPro", "Jenis Produk Rata-rata"),
               ("totKat", "Kategori Produk Rata-rata")]

    for i, (colname, title) in enumerate(metrics):
        fig = go.Figure()
        fig.add_trace(go.Indicator(mode="number", value=groupByCustomer[colname].mean(), title={"text": title}))
        fig.update_layout(height=150)
        cols[i].plotly_chart(fig, use_container_width=True)

# Chatbot Page
elif st.session_state.page == "Chatbot":
    st.title("🤖 Simple Chatbot with Adashboard")
    client = ollama.Client()
    model = "granite3-dense:2b"

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hi! How can I help you today?"}]
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something about your data...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = client.generate(model=model, prompt=user_input)
                st.markdown(response.response)
        st.session_state.messages.append({"role": "assistant", "content": response.response})
