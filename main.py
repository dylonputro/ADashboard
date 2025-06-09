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
# Set background merah
st.markdown(
    """
    <style>
    .stApp {
        background-color: #808080;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# === Session State Setup ===
if "page" not in st.session_state:
    st.session_state.page = "Upload Data"
if "df" not in st.session_state:
    st.session_state.df = None
if "column_mapping" not in st.session_state:
    st.session_state.column_mapping = {}
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you today?"}]

# === Sidebar Navigation ===
with st.sidebar:
    st.title("Navigation")
    st.markdown("Select a section below:")
    page = st.radio("Menu", ["Upload Data", "Dashboard", "Chatbot"], index=["Upload Data", "Dashboard", "Chatbot"].index(st.session_state.page))
    st.session_state.page = page

    if st.session_state.df is not None:
        st.divider()
        st.markdown("### Dataset Info")
        st.write("✅ Rows:", st.session_state.df.shape[0])
        st.write("✅ Columns:", st.session_state.df.shape[1])

# === Functions ===
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# === Page: Upload Data ===
if st.session_state.page == "Upload Data":
    st.title("📤 Upload Transaction Data")
    st.markdown("Please upload your CSV file below:")
    uppath = st.file_uploader("Drop your file here (.csv)", type="csv")

    if uppath:
        try:
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
                    st.rerun()

            if st.button("Proceed to Dashboard"):
                st.session_state.df = prepro.clean_data(st.session_state.df)
                st.session_state.page = "Dashboard"
                st.rerun()
        except Exception as e:
            st.error(f"Error loading file: {e}")

# === Page: Dashboard ===
elif st.session_state.page == "Dashboard":
    if st.session_state.df is None:
        st.warning("Please upload the data first!")
    else:
        st.title("📊 Sales & Customer Dashboard")

        salesVsTime = prepro.prep_sales(st.session_state.df)
        groupByCustomer = prepro.prep_customer(st.session_state.df)
        groupByHour = prepro.prep_grouphour(st.session_state.df)
        groupByProduct = prepro.prep_groupProduct(st.session_state.df)
        groupByKategori = prepro.prep_groupKategori(st.session_state.df)

        # Sales Dashboard
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="number+delta",
                    value=salesVsTime['nominal_transaksi'].mean(),
                    title={"text": "Rata-Rata Pemasukan Harian"},
                    delta={"reference": salesVsTime['nominal_transaksi'].mean() - (salesVsTime["nominal_transaksi"].iloc[-1] - salesVsTime["nominal_transaksi"].iloc[-2]),
                           "relative": False,
                           "increasing": {"color": "green"},
                           "decreasing": {"color": "red"}},
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
                    delta={"reference": salesVsTime['banyak_produk'].mean() - (salesVsTime["banyak_produk"].iloc[-1] - salesVsTime["banyak_produk"].iloc[-2]),
                           "relative": False,
                           "increasing": {"color": "green"},
                           "decreasing": {"color": "red"}},
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
                    delta={"reference": salesVsTime['banyak_transaksi'].mean() - (salesVsTime["banyak_transaksi"].iloc[-1] - salesVsTime["banyak_transaksi"].iloc[-2]),
                           "relative": False,
                           "increasing": {"color": "green"},
                           "decreasing": {"color": "red"}},
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
                try:
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
                        name='Actual',
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
                except Exception as e:
                    st.error(f"Error loading forecast model or plotting: {e}")

        # Product Dashboard
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
                fig = px.line(st.session_state.df, x="Tanggal & Waktu", y="Jumlah Produk", color="Kategori", title="Grafik Jumlah Produk per Kategori")
                st.plotly_chart(fig)

        with st.container():
            st.subheader("Banyak Produk per Kategori")
            fig = px.bar(groupByKategori, x="Kategori", y="Jumlah_produk", color="Kategori")
            st.plotly_chart(fig)

        with st.container():
            st.subheader("Rata-rata Nominal Transaksi per Tipe Penjualan")
            fig = px.bar(
                st.session_state.df.groupby("Tipe Penjualan")["Harga Produk"].mean().reset_index(),
                x="Tipe Penjualan", y="Harga Produk",
                color="Tipe Penjualan"
            )
            st.plotly_chart(fig)

        if st.button("Back to Upload"):
            st.session_state.page = "Upload Data"
            st.rerun()

# === Page: Chatbot ===
elif st.session_state.page == "Chatbot":
    st.title("💬 Chatbot - Ollama GPT")

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            st.markdown(f"**Assistant:** {message['content']}")
        else:
            st.markdown(f"**You:** {message['content']}")

    user_input = st.text_input("Ask me anything...")

    if st.button("Send") and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Generate response
        try:
            response = ollama.ask(user_input)
        except Exception as e:
            response = f"Error: {e}"
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
