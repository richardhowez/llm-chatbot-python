import time
import streamlit as st
from utils import write_message
from llm import llm, embeddings
# from graph import graph
from agent import generate_response
from predict_page import show_predict_page
from explore_page import show_explore_page
from chatQuery import start_chat
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
# tag::setup[]

# Page Config
#st.set_page_config("Ebert", page_icon=":movie_camera:")
 # end::setup[]
# Set up Session State
page = "Explore"
page = st.sidebar.selectbox("Explore, Predict or Query", ("Predict", "Explore", "Query"))

if page == "Predict":
    show_predict_page()
if page == "Explore":
    show_explore_page()
if page == "Query":
    start_chat()
#else :
    #st.empty()
    #st.balloons()
    #st.progress(10)
    #with st.spinner('Wait for it...'):    time.sleep(3)
