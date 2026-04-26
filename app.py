import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load lightweight model (best for your system)
model_name = "microsoft/DialoGPT-small"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

st.title("🤖 AI Chatbot (No API)")
st.write("Ask me anything!")

# Session state
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input
user_input = st.chat_input("Type your message...")

if user_input:
    st.chat_message("user").write(user_input)
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
        max_length=500,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    # Decode response
    response = tokenizer.decode(
        st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    st.chat_message("assistant").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
