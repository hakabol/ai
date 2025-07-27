from model import chatbot_assistance
import streamlit as st

st.title("AI")

assistant = chatbot_assistance()
assistant.load_settings("settings.py")
print(assistant.intents_path)
assistant.pass_intents()
assistant.prepare_data()
assistant.load("chatbot_model.pth", "dimensions.json")

if 'message' not in st.session_state:
    st.session_state.message = []

message = st.chat_input()

for mess in st.session_state.message:
    st.chat_message(mess["role"]).markdown(mess["content"])

if message:
    st.chat_message("user").markdown(message)
    st.session_state.message.append({"role":"user", "content":message})
    output = assistant.process_message(message)
    st.chat_message("ai").markdown(output)
    st.session_state.message.append({"role":"ai", "content":output})
