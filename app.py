import streamlit as st

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

st.title("🤖 AI Chatbot")
st.write("Simple and stable chatbot (no crash)")

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input
user_input = st.chat_input("Type your message...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    text = user_input.lower()

    # RULE-BASED LOGIC (SMART + CLEAN)
    if text in ["hi", "hello", "hey"]:
        response = "Hello! How can I help you today? 😊"

    elif "how are you" in text:
        response = "I'm doing great! Thanks for asking 😊"

    elif "your name" in text:
        response = "I'm your AI chatbot 🤖"

    elif "bye" in text:
        response = "Goodbye! Have a nice day 👋"

    elif "help" in text:
        response = "I can chat with you, answer basic questions, and assist you 😊"

    else:
        response = "That's interesting! Tell me more 😊"

    st.chat_message("assistant").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
