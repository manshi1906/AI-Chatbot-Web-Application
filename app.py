import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load lightweight model
model_name = "microsoft/DialoGPT-small"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# UI
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot")
st.write("General conversation assistant")

# Session state
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Encode input
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Add history
    bot_input_ids = torch.cat(
        [st.session_state.chat_history_ids, new_input_ids],
        dim=-1
    ) if st.session_state.chat_history_ids is not None else new_input_ids

    # Generate response
    st.session_state.chat_history_ids = model.generate(
        bot_input_ids,
        max_length=300,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=40,
        top_p=0.9,
        temperature=0.7
    )

    # Decode response
    response = tokenizer.decode(
        st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    # 🔧 CLEAN RESPONSE
    response = response.replace("User:", "").replace("Assistant:", "").strip()

    # Remove garbage / very short replies
    if len(response) < 2:
        response = "I'm here! How can I help you? 😊"

    # Remove unwanted topics
    bad_words = ["pokemon", "trade", "friend code"]
    if any(word in response.lower() for word in bad_words):
        response = "Let's talk about something else 😊"

    # Limit long responses
    if len(response) > 200:
        response = response[:200] + "..."

    # Show bot response
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
