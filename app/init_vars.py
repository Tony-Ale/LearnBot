import streamlit as st
import queue
def rag_vars():
    st.session_state.rag_queue = queue.Queue()
    st.session_state.run_if_no_pipeline = True
    st.session_state.rag_run_every = 2
    st.session_state.disable_rag_chat_input = True

def mcq_vars():
    st.session_state.mcq_queue = queue.Queue()
    st.session_state.run_every = 2
    st.session_state.run_if_no_data = True
    st.session_state.disable_quiz_button = True

def video_result_vars():
    st.session_state.video_result_queue = queue.Queue()
    st.session_state.video_run_every = 2
    st.session_state.run_if_no_video_result = True
    st.session_state.disable_video_button = True

def url_vars():
    st.session_state.url = []
    st.session_state.video_title = []

def error_handling_vars():
    st.session_state.is_error = False
    st.session_state.error_type = None 
    st.session_state.activate_error_dialogue = False

def video_index_vars():
    # Initialize the video index in session state
    st.session_state.video_index = 0

def quiz_questions_vars():
    st.session_state.quiz_questions = []
    st.session_state.question_index = 0

def init_go_button_vars():
    st.session_state.disable_go_button_count = 0
    st.session_state.disable_go_button = False
    st.session_state.is_go_button_clicked = False

def initialise_variables():
    if 'rag_queue' not in st.session_state:
        rag_vars()
    if 'mcq_queue' not in st.session_state:
        mcq_vars()
    if 'video_result_queue' not in st.session_state:
        video_result_vars()
    if "url" not in st.session_state:
        url_vars()
    if "error_type" not in st.session_state:
        error_handling_vars()
    if 'video_index' not in st.session_state:
        video_index_vars()
    if 'quiz_questions' not in st.session_state:
        quiz_questions_vars()
    if "disable_go_button" not in st.session_state:
        init_go_button_vars()