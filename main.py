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

# Set page configuration
st.set_page_config(layout="wide", page_title="Dashboard Group 15", page_icon="📊")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f5;
    }
    .title {
        font-size: 2.5em;
        color: #1F2A44;
    }
    .indicator {
        font-size: 60px;
        color: #1F2A44;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("📊 Dashboard By Group 15")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Upload Data", "Dashboard"])

# Function to read data
def load_data(path: str):
    data = pd.read_csv(path)
    return data 

# Initialize session state variables
if "df" not in st.session_state:
    st.session_state.df = None
if "column_mapping" not in st.session_state:
    st.session_state.column_mapping = {}

if page == "Upload Data":
    # Drag and drop feature
    uppath = st.file_uploader("Drop your file please")
    if uppath is not None:  
        try:
            st.session_state.df = load_data(uppath)
            if st.session_state.df is not None:
                with st.expander("Data Preview"):  
                    st.dataframe(st.session_state.df)
                    st.write(st.session_state.df.dtypes)
                standard_columns = ['Tanggal & Waktu', 'ID Struk', 'Tipe Penjualan', 'Nama Pelanggan','Nama Produk', 'Kategori', 'Jumlah Produk', 'Harga Produk', 'Metode Pembayaran']
                submitted_columns = st.session_state.df.columns.tolist()
                # Standardize column names 
                if set(submitted_columns) == set(standard_columns):
                    st.success("✅ Column names match the standard.")
                else:
                    st.warning("⚠️ Column names do not match the standard!")
                    for i, col in enumerate(standard_columns):
                        default_value = st.session_state.column_mapping.get(col, None)  # Get previous value if exists
                        st.session_state.column_mapping[col] = st.selectbox(
                            f"Select column for '{col}'", submitted_columns, index=submitted_columns.index(default_value) if default_value in submitted_columns else 0, key=f"col_{i}"
                        )
                    if st.button("Change"):
                        st.session_state.df = prepro.fix_column_name(st.session_state.df, st.session_state.column_mapping)
                if st.button("Continue"):
                    if st.session_state.df is not None:
                        st.session_state.df = prepro.clean_data(st.session_state.df)
                        st.experimental_rerun()
        except Exception as e:
            st.error(f"An error occurred while loading the data: {e}")

elif page == "Dashboard":
    if st.session_state.df is not None:
        try:
            salesVsTime = prepro.prep_sales(st.session_state.df)
            groupByCustomer = prepro.prep_customer(st.session_state.df)
            groupByHour = prepro.prep_grouphour(st.session_state.df)
            groupByProduct = prepro.prep_groupProduct(st.session_state.df)
            groupByKategori = prepro.prep_groupKategori(st.session_state.df)

            # Sales Dashboard
            st.header("Sales Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="number+delta",
                    value=salesVsTime['nominal_transaksi'].mean(),
                    title={"text": "Rata-Rata Pemasukan Harian"},
                    delta={"reference": salesVsTime['nominal_transaksi'].mean() - (salesVsTime["nominal_transaksi"].iloc[-1] - salesVsTime["nominal_transaksi"].iloc[-2]), "relative": False, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
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
                    delta={"reference": salesVsTime['banyak_produk'].mean() - (salesVsTime["banyak_produk"].iloc[-1] - salesVsTime["banyak_produk"].iloc[-2]), "relative": False, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
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
                    delta={"reference": salesVsTime['banyak_transaksi'].mean() - (salesVsTime["banyak_transaksi"].iloc[-1] - salesVsTime["banyak_transaksi"].iloc[-2]), "relative": False, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
                    number={"font": {"size": 60, "color": "#1F2A44"}}
                ))
                fig.update_layout(width=400, height=150)
                st.plotly_chart(fig, use_container_width=True)

            # Additional visualizations
            st.subheader("Sales Trends")
            col1, col2 = st.columns(2)
            with col1:
                fig = px.line(salesVsTime, x="Tanggal & Waktu", y="banyak_transaksi", title="Banyak Transaksi Seiring Waktu")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.line(groupByHour, x="Jam", y="Jumlah_produk", title="Rata-rata Banyak Produk yang dipesan dalam Seharian")
                st.plotly_chart(fig, use_container_width=True)

            # Product Dashboard
            st.subheader("Product Insights")
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

            # Customer Segmentation Dashboard
            st.subheader("Customer Segmentation")
            groupByCustomer = prepro.customer_segmentation(groupByCustomer)
            valueCCount = groupByCustomer["cluster"].value_counts().reset_index()
            valueCCount.columns = ["cluster", "count"]
            fig = px.bar(valueCCount, x="cluster", y="count", color="cluster", title="Bar Chart Jumlah Produk per Kategori")
            st.plotly_chart(fig)

            # Chatbot Section
            st.title("🤖 Simple Chatbot with Adashboard")
            client = ollama.Client()
            model = "granite3-dense:2b"

            if "messages" not in st.session_state:
                st.session_state["messages"] = [{"role": "assistant", "content": "Hi! How can I help you today?"}]
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            user_input = st.chat_input("Type your message...")
            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = client.generate(model=model, prompt=(user_input))
                        st.markdown(response.response)
                st.session_state.messages.append({"role": "assistant", "content": response.response})
        except Exception as e:
            st.error(f"An error occurred while processing the dashboard: {e}")
    else:
        st.warning("Please upload data to proceed.")
