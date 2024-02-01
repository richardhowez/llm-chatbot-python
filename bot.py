import time
import streamlit as st
from utils import write_message
from llm import llm, embeddings
# from graph import graph
from agent import generate_response
from predict_page import show_predict_page
from explore_page import show_explore_page
from chatQuery import start_chat
# tag::setup[]

# Page Config
#st.set_page_config("Ebert", page_icon=":movie_camera:")
 # end::setup[]
# Set up Session State
page = "Explore"
page = st.sidebar.selectbox("Explore, Predict or Query", ("Predict", "Explore", "Query"))

if page == "Predict":
    #show_predict_page()
    show_explore_page()
if page == "Explore":
    show_explore_page()
if page == "Query":
    start_chat()
#else :
    #st.empty()
    #st.balloons()
    #st.progress(10)
    #with st.spinner('Wait for it...'):    time.sleep(3)
