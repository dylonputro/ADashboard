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
