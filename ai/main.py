from model import chatbot_assistance
import streamlit as st
import os
import time

st.title("AI")

if "assistant" not in st.session_state:
    current_dir = os.path.dirname(__file__)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    settings_path = os.path.join(current_dir, "settings.json")

    st.session_state.assistant = chatbot_assistance()
    st.session_state.assistant.load_settings(settings_path)

    st.session_state.assistant.pass_intents()
    st.session_state.assistant.prepare_data()
    model_path = os.path.join(current_dir, "chatbot_model.pth")
    dimensions_path = os.path.join(current_dir, "dimensions.json")
    #st.write("Current dir:", current_dir)
    #st.write("intents path:", "chatbot_model.pth")
    #st.write("File exists:", model_path)
    st.session_state.assistant.load(model_path, dimensions_path)

if 'message' not in st.session_state:
    st.session_state.message = []

message = st.chat_input()

for mess in st.session_state.message:
    st.chat_message(mess["role"]).markdown(mess["content"])

if message:
    st.chat_message("user").markdown(message)
    st.session_state.message.append({"role":"user", "content":message})
    #st.write(f"model input: {len(assistant.bag_of_words(assistant.token_lemon(message)))}, input input: {len((assistant.vocaluberries))}")
    output = st.session_state.assistant.process_message(message)
    st.chat_message("ai").markdown(output)
    st.session_state.message.append({"role":"ai", "content":output})
