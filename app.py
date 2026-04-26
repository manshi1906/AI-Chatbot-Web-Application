import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

chatbot = load_model()

st.title("🤖 Smart Chatbot (No API)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Better prompting
    prompt = f"Answer like a helpful assistant: {user_input}"

    response = chatbot(prompt, max_length=100, do_sample=True)[0]["generated_text"]

    st.chat_message("assistant").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
