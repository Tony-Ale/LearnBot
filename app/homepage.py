import streamlit as st
import time
from core import Data_Store

# Center text using HTML and Streamlit's markdown function
#homepage_text = "Learn Beyond Limits: Instant Courses, Interactive Quizzes, and an AI Assistant to Chat with Your Videos and Documents."
#st.markdown(f"<h1 style='text-align: center;'>{homepage_text}</h1>", unsafe_allow_html=True)

homepage_text = "Learn Beyond Limits: Instant Courses, Interactive Quizzes, and an AI Assistant to Chat with Your Videos and Documents."

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600&display=swap');

    .centered-text {
        font-family: 'Roboto Mono', monospace;
        font-size: 2em;
        text-align: center;
        color: white;
        line-height: 1.5;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#st.markdown(f"<div class='centered-text'>{homepage_text}</div>", unsafe_allow_html=True)
if "is_api_key" not in st.session_state:
    st.session_state.is_api_key = False

if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False
    
@st.dialog("Insert Cohere API key")
def get_api_key():
    api_key = st.text_input(label="API key", type="password")

    if st.button("Submit"):
        with st.spinner("Inserting API key"):
            if api_key:
                st.session_state.is_api_key = True
                Data_Store.pass_api_key_to_llm(key=api_key)
                st.rerun()


def on_click():
    st.session_state.button_clicked = True


_, main_col, _ = st.columns([0.10, 0.80, 0.10])

with main_col:
    st.markdown(f"<div class='centered-text'>{homepage_text}</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([0.90, 0.10])
    with col1:
        st.session_state.user_input = st.text_input("input anything you want to learn", label_visibility="collapsed", placeholder="input anything you want to learn")

    with col2:

        st.button('Go', on_click=on_click)

        if st.session_state.button_clicked:
            if not st.session_state.is_api_key:
                get_api_key()
            if st.session_state.user_input and st.session_state.is_api_key:
                st.session_state.button_clicked = False
                with st.spinner("Getting videos"):
                    st.session_state.switch_from_homepage_to_learning_page = True
                    st.switch_page(st.session_state.learning_page)

