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

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "upload"
if "df" not in st.session_state:
    st.session_state.df = None
if "column_mapping" not in st.session_state:
    st.session_state.column_mapping = {}

# Sidebar for navigation
st.sidebar.title("Menu")
page = st.sidebar.radio("Pilih halaman:", ["Upload Data", "Dashboard"])

st.session_state.page = "upload" if page == "Upload Data" else "Dashboard"

if st.session_state.page == "upload":
    st.sidebar.subheader("Upload File CSV")
    uppath = st.sidebar.file_uploader("Drop your file please")
    if uppath is not None:
        st.session_state.df = load_data(uppath)
    
    if st.session_state.df is not None:
        with st.expander("Data Preview"):  
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
                st.session_state.column_mapping[col] = st.sidebar.selectbox(
                    f"Select column for '{col}'", submitted_columns,
                    index=submitted_columns.index(default_value) if default_value in submitted_columns else 0,
                    key=f"col_{i}"
                )
            if st.sidebar.button("Change Column Mapping"):
                st.session_state.df = prepro.fix_column_name(st.session_state.df, st.session_state.column_mapping)
        
        if st.sidebar.button("Continue to Dashboard"):
            st.session_state.page = "Dashboard"
            st.session_state.df = prepro.clean_data(st.session_state.df)
            st.experimental_rerun()

elif st.session_state.page == "Dashboard":
    if st.session_state.df is None:
        st.warning("Data belum diupload. Silakan upload data di halaman Upload Data terlebih dahulu.")
        st.stop()

    # Sidebar filter for dashboard
    st.sidebar.subheader("Filter Dashboard")
    groupByCustomer = prepro.prep_customer(st.session_state.df)
    optionCluster = ["All"] + groupByCustomer["cluster"].unique().tolist()
    selected_cluster = st.sidebar.selectbox("Pilih Cluster Customer", optionCluster)

    salesVsTime = prepro.prep_sales(st.session_state.df)
    groupByHour = prepro.prep_grouphour(st.session_state.df)
    groupByProduct = prepro.prep_groupProduct(st.session_state.df)
    groupByKategori = prepro.prep_groupKategori(st.session_state.df)

    # Sales Dashboard
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])
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
    
    # Example cluster filtering
    groupByCustomer = prepro.customer_segmentation(groupByCustomer)
    if selected_cluster == "All":
        clusteringmask = groupByCustomer.copy()
    else:
        clusteringmask = groupByCustomer[groupByCustomer["cluster"] == selected_cluster].copy().reset_index()
    
    # Customer segmentation dashboard (as before)...

    # (Selanjutnya lanjutkan sesuai kode Anda...)

    # Anda bisa memasukkan chatbox dan lainnya di bagian bawah halaman dashboard

    # Contoh chatbox tetap sama

    with st.container(): 
        st.title("🤖 Simple Chatbot with Adashboard")
        client = ollama.Client()
        model  = "granite3-dense:2b"

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
