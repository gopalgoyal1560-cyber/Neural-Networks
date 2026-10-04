import requests
import streamlit as st

API_URL = "https://neural-networks-wked.onrender.com/Post_values"

st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("🤖 IQ of A, Explanation like K. Project 5 result failure!.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(API_URL, json={"Value": prompt}, timeout=90)
                response.raise_for_status()
                data = response.json()
                reply = data.get("response", str(data)) if isinstance(data, dict) else str(data)
            except Exception as error:
                reply = f"Error: {error}"
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})