import streamlit as st
import os

def construct_path(file_name:str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, file_name)
    return file_path

def learn_bot():
    st.set_page_config(layout="wide")
    homepage_path = construct_path("homepage.py")
    learning_interface_path = construct_path("learning_interface.py")

    st.session_state.homepage = st.Page(homepage_path, title="Home")
    st.session_state.learning_page = st.Page(learning_interface_path, title="LearnBot")
    pg = st.navigation([st.session_state.homepage, st.session_state.learning_page], position="hidden")

    pg.run()

if __name__ == '__main__':
    learn_bot()