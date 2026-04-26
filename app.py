import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/DialoGPT-small"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

st.title("🤖 AI Chatbot")
st.write("General conversation assistant")

if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Add guiding prompt
    prompt = "You are a helpful and friendly assistant. Answer clearly and professionally.\nUser: " + user_input

    new_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')

    bot_input_ids = torch.cat(
        [st.session_state.chat_history_ids, new_input_ids],
        dim=-1
    ) if st.session_state.chat_history_ids is not None else new_input_ids

    st.session_state.chat_history_ids = model.generate(
        bot_input_ids,
        max_length=400,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=40,
        top_p=0.9,
        temperature=0.6   # less random
    )

    response = tokenizer.decode(
        st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    # Clean weird outputs
    if len(response) > 200:
        response = response[:200] + "..."

    if any(word in response.lower() for word in ["pokemon", "trade", "friend code"]):
        response = "Let's talk about something else. How can I help you?"

    st.chat_message("assistant").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
