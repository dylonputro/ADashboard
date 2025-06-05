import pandas as pd 
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import prepro
import ollama
import joblib 
from darts.models import NBEATSModel

st.set_page_config(layout="wide", page_title="Dashboard Group 15", page_icon="📊")

# Sidebar Navigation
st.sidebar.title("📊 Menu Dashboard")
page = st.sidebar.radio("Pilih Halaman:", ["Upload Data", "Dashboard", "Chatbot"])

# Fungsi read data 
@st.cache_data
def load_data(path: str):
    data = pd.read_csv(path)
    return data

# Init session state
if "df" not in st.session_state:
    st.session_state.df = None
if "column_mapping" not in st.session_state:
    st.session_state.column_mapping = {}

# --- Upload Page ---
if page == "Upload Data":
    st.title("📥 Upload Data Penjualan")
    uploaded_file = st.file_uploader("Seret dan jatuhkan file CSV Anda di sini", type=["csv"])
    if uploaded_file:
        st.session_state.df = load_data(uploaded_file)
        st.success("File berhasil diupload!")
        with st.expander("Preview Data"):
            st.dataframe(st.session_state.df, use_container_width=True)
            st.write("Tipe Data Kolom:")
            st.write(st.session_state.df.dtypes)

        standard_columns = ['Tanggal & Waktu', 'ID Struk', 'Tipe Penjualan', 'Nama Pelanggan',
                            'Nama Produk', 'Kategori', 'Jumlah Produk', 'Harga Produk', 'Metode Pembayaran']
        submitted_columns = st.session_state.df.columns.tolist()

        if set(submitted_columns) == set(standard_columns):
            st.success("✅ Nama kolom sudah sesuai standar.")
        else:
            st.warning("⚠️ Nama kolom tidak sesuai standar! Silakan mapping kolom berikut:")
            for i, col in enumerate(standard_columns):
                default_value = st.session_state.column_mapping.get(col, None)
                st.session_state.column_mapping[col] = st.selectbox(
                    f"Mapping untuk '{col}'", submitted_columns,
                    index=submitted_columns.index(default_value) if default_value in submitted_columns else 0,
                    key=f"col_{i}"
                )
            if st.button("Terapkan Mapping"):
                st.session_state.df = prepro.fix_column_name(st.session_state.df, st.session_state.column_mapping)
                st.experimental_rerun()

        if st.button("Lanjut ke Dashboard"):
            st.session_state.df = prepro.clean_data(st.session_state.df)
            st.experimental_rerun()

# --- Dashboard Page ---
elif page == "Dashboard" and st.session_state.df is not None:
    st.title("📈 Dashboard Penjualan")
    
    salesVsTime = prepro.prep_sales(st.session_state.df)
    groupByCustomer = prepro.prep_customer(st.session_state.df)
    groupByHour = prepro.prep_grouphour(st.session_state.df)
    groupByProduct = prepro.prep_groupProduct(st.session_state.df)
    groupByKategori = prepro.prep_groupKategori(st.session_state.df)
    
    st.subheader("📊 Ringkasan Penjualan Harian")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rata-rata Pemasukan Harian", f"Rp {salesVsTime['nominal_transaksi'].mean():,.0f}")
    col2.metric("Rata-rata Produk Harian", f"{salesVsTime['banyak_produk'].mean():,.0f}")
    col3.metric("Rata-rata Transaksi Harian", f"{salesVsTime['banyak_transaksi'].mean():,.0f}")

    st.divider()
    
    st.subheader("📅 Tren Penjualan dan Produk")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.line(salesVsTime, x="Tanggal & Waktu", y="banyak_transaksi", title="Banyak Transaksi Seiring Waktu")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.line(groupByHour, x="Jam", y="Jumlah_produk", title="Rata-rata Produk per Jam")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("📦 Produk Populer dan Kategori")
    col1, col2 = st.columns(2)
    with col1:
        top_5 = groupByProduct.nlargest(8, "Jumlah_produk")
        other_total = groupByProduct.loc[~groupByProduct["Nama Produk"].isin(top_5["Nama Produk"]), "Jumlah_produk"].sum()
        other_row = pd.DataFrame([{"Nama Produk": "Other", "Jumlah_produk": other_total}])
        top_5 = pd.concat([top_5, other_row], ignore_index=True)
        fig3 = px.pie(top_5, names="Nama Produk", values="Jumlah_produk", hole=0.4, title="Donat Produk Terlaris")
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        fig4 = px.bar(groupByKategori, x="Kategori", y="Total_omset", color="Kategori", title="Omset per Kategori")
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    
    st.subheader("👥 Segmentasi Pelanggan")
    groupByCustomer = prepro.customer_segmentation(groupByCustomer)
    valueCCount = groupByCustomer["cluster"].value_counts().reset_index()
    valueCCount.columns = ["cluster", "count"]
    fig5 = px.bar(valueCCount, x="cluster", y="count", color="cluster", title="Jumlah Pelanggan per Klaster")
    st.plotly_chart(fig5, use_container_width=True)

    optionCluster = ["All"] + groupByCustomer["cluster"].unique().tolist()
    selected_cluster = st.selectbox("Pilih Klaster Pelanggan", optionCluster)
    if selected_cluster == "All":
        clusteringmask = groupByCustomer.copy()
    else:
        clusteringmask = groupByCustomer[groupByCustomer["cluster"] == selected_cluster].copy().reset_index()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rata-Rata Pengeluaran", f"Rp {clusteringmask['totSpen'].mean():,.0f}")
    col2.metric("Rata-Rata Jumlah Produk", f"{clusteringmask['totJum'].mean():,.0f}")
    col3.metric("Rata-Rata Jumlah Jenis Produk", f"{clusteringmask['totJenPro'].mean():,.0f}")
    col4.metric("Rata-Rata Jumlah Kategori Pesanan", f"{clusteringmask['totKat'].mean():,.0f}")

# --- Chatbot Page ---
elif page == "Chatbot":
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

else:
    st.warning("Silakan upload data terlebih dahulu pada menu Upload Data")

