import streamlit as st
from utils import write_message
from llm import llm, embeddings
# from graph import graph
from agent import generate_response

# tag::submit[]
# Submit handler
def handle_submit(message):
    # Handle the response
    with st.spinner('Thinking...'):
        response = generate_response(message)
        write_message('assistant', response)
    # end::submit[]
                
def start_chat():
    st.empty()
    if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi, I'm the GraphAcademy Chatbot!  How can I help you?"},
            ]
        # end::session[]
    # tag::chat[]
    # Display messages in Session State
    for message in st.session_state.messages:
         write_message(message['role'], message['content'], save=False)

    # Handle any user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        write_message('user', prompt)

        # Generate a response
        handle_submit(prompt)
    # end::chat[]