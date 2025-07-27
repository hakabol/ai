import streamlit as st
import os
from model import chatbot_assistance

st.title("💬 Simple AI Chatbot")

@st.cache_resource
def load_bot():
    assistant = chatbot_assistance()
    base = os.path.dirname(__file__)
    assistant.load_settings(os.path.join(base, "settings.py"))
    assistant.pass_intents()
    assistant.load(os.path.join(base, "chatbot_model.pth"), os.path.join(base, "dimensions.json"))
    return assistant

bot = load_bot()

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.chat_input("Say something...")

for msg in st.session_state.chat:
    st.chat_message(msg["role"]).markdown(msg["content"])

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat.append({"role": "user", "content": user_input})
    reply = bot.process_message(user_input)
    st.chat_message("ai").markdown(reply)
    st.session_state.chat.append({"role": "ai", "content": reply})
