import streamlit as st
from core import get_videos, generate_mcqs, Data_Store, RAG
from app.interface_funcs import *
from app.errors import ErrorType, ERROR_LIST
from cohere.core.api_error import ApiError
from app.init_vars import *
from app.background_tasks import *
import queue
import time
import uuid

#st.set_page_config(layout="wide")
#-------------------------------------Initialize session state---------------------------------#

initialise_variables()

#-------------------------------------functions---------------------------------#
def enable_go_button():
    if st.session_state.disable_go_button_count >= 2:
        st.session_state.disable_go_button = False

@st.dialog("Error")
def error_dialogue(error):
    if error == ErrorType.API_KEY_ERROR:
        st.write("Insert a valid api key")
    elif error == ErrorType.OTHER_ERRORS:
        st.write("Poor internet connection")
    else:
        st.write(error)

def handle_error(error:ErrorType|str):
    st.session_state.is_error = True
    st.session_state.activate_error_dialogue = True
    if st.session_state.error_type is None:
        st.session_state.error_type = error


@st.fragment(run_every=st.session_state.run_every)
def get_quiz_question_if_ready():
    if st.session_state.run_if_no_data:

        output:queue.Queue = st.session_state.mcq_queue
        if not output.empty():

            queue_result = output.get_nowait()

            if queue_result in ERROR_LIST or not queue_result:
                if not queue_result:
                    handle_error(error="There is no Data to generate quizzes")
                else:
                    handle_error(queue_result)

                st.session_state.disable_go_button = False

            else:
                st.session_state.quiz_questions = queue_result

                st.session_state.disable_quiz_button = False 
                st.session_state.disable_go_button_count += 1
                enable_go_button()

            st.session_state.run_if_no_data = False
            st.session_state.run_every = None
            st.rerun()

@st.fragment(run_every=st.session_state.rag_run_every)
def get_rag_pipeline_if_ready():
    if st.session_state.run_if_no_pipeline:

        output:queue.Queue = st.session_state.rag_queue
        if not output.empty():
            queue_result = output.get_nowait()

            if queue_result in ERROR_LIST:
                handle_error(queue_result)
                
                st.session_state.disable_go_button = False

            else:
                st.session_state.vectordb, st.session_state.conversation_chain = queue_result

                st.session_state.disable_go_button_count += 1
                st.session_state.disable_rag_chat_input = False
                enable_go_button()
                st.toast("The Doc bot is ready!")
                time.sleep(1)

            st.session_state.run_if_no_pipeline = False
            st.session_state.rag_run_every = None
            st.rerun()

@st.fragment(run_every=st.session_state.video_run_every)
def get_video_result_if_ready():
    if st.session_state.run_if_no_video_result:

        output:queue.Queue = st.session_state.video_result_queue

        if not output.empty():
            queue_result = output.get_nowait()

            if queue_result in ERROR_LIST:
                handle_error(queue_result)
                st.session_state.disable_go_button = False

            else:
                st.session_state.update(queue_result)
                get_mcq_and_prepare_rag_in_background()

                st.session_state.disable_video_button = False
                st.session_state.is_go_button_clicked = False

            st.session_state.run_if_no_video_result = False
            st.session_state.video_run_every = None
            st.rerun()


def set_quiz():
    quiz_data = st.session_state.quiz_questions[st.session_state.question_index]
    st.write(quiz_data['llm_generated_query'])

    with st.container(height=420):
        user_answer = st.radio(
                    label=quiz_data['question'],
                    options= quiz_data['options'],
                    index=None,
                )
    correct_answer = quiz_data['answer']

    if user_answer == correct_answer:
        st.write('correct')
    elif user_answer is None:
        st.write('select an answer')
    else:
        st.write('wrong answer')

    prev_quiz_button_col, next_quiz_button_col = st.columns([0.85, 0.15])
    with prev_quiz_button_col:
        if st.button("Previous", on_click=prev_question, disabled=st.session_state.disable_quiz_button):
            pass
    
    with next_quiz_button_col:
        if st.button("Next", on_click=next_question, disabled=st.session_state.disable_quiz_button):
            pass
    
    change_radio_button_font_size(quiz_data['question'], size=28, margin_top=0, paddingBottom=10)
    for label in quiz_data['options']:
        change_radio_button_font_size(label, size=22, margin_top=-5, paddingBottom=10)
    
#------------------------------------Streamlit code--------------------------------#

text_box_col, go_button_col = st.columns([0.9, 0.1])

with text_box_col:
    user_input = st.text_input(label="Input what you want to learn", label_visibility="collapsed", placeholder="Input what you want to learn")
    if st.session_state.switch_from_homepage_to_learning_page and not user_input:
        reset_vars()
        user_input = st.session_state.user_input


with go_button_col:
    if (st.button("Go", on_click=reset_vars, disabled=st.session_state.disable_go_button) or st.session_state.switch_from_homepage_to_learning_page) and user_input:
        st.session_state.switch_from_homepage_to_learning_page = False
        with st.spinner("Getting videos"):
            start = time.time()

            run_in_background(task=get_videos_in_background,
                              kwargs={'user_input':user_input,
                                      'activate_data_store':True,
                                      'output':st.session_state.video_result_queue})

            # Initialise RAG Pipeline with an empty string
            try:
                if "vectordb" not in st.session_state:
                    st.session_state.vectordb = RAG.gen_faiss(splits=[" "],
                                                            ids=None)
                else:
                    if "rag_ids" in st.session_state:
                        pass
                        #RAG.delete_embedding_in_faiss_vectordb(vectordb=st.session_state.vectordb,
                                                            #ids=st.session_state.rag_ids)
            except ApiError as e:
                if e.status_code == 401:
                    handle_error(ErrorType.API_KEY_ERROR)
                else:
                    handle_error(ErrorType.OTHER_ERRORS)

                st.session_state.is_go_button_clicked = False
            except:
                handle_error(ErrorType.OTHER_ERRORS)
                st.session_state.is_go_button_clicked = False

            end = time.time()

video_col, quiz_rag_col = st.columns([0.6, 0.4])

with video_col:

    #if st.session_state.url:
    if st.session_state.disable_video_button:
        if st.session_state.is_error:
            with st.chat_message("AI"):
                st.write("An Error occurred while fetching videos")
        else:
            with st.chat_message("AI"):
                st.write("I'm fetching the videos. This may take a few moments")
    else:
        render_video(st.session_state.video_index)  

    prev_video_button_col, next_video_button_col = st.columns([0.85, 0.15])
    with prev_video_button_col:
        if st.button("Previous", on_click=prev_video, key='previous_video_button', disabled=st.session_state.disable_video_button):
            pass
    
    with next_video_button_col:
        if st.button("Next", on_click=next_video, key='next_video_button', disabled=st.session_state.disable_video_button):
            pass

with quiz_rag_col:
    quiz_tab, rag_tab = st.tabs(['Quiz', 'RAG'])
    with quiz_tab:
        
        if st.session_state.disable_quiz_button:
            if st.session_state.is_error:
                with st.chat_message("AI"):
                    st.write("An Error occurred while generating quizzes")
            else:
                with st.chat_message("AI"):
                    st.write("I'm generating the quizzes. This may take a few moments")
        else:
            set_quiz()

    with rag_tab:

        container = st.container(height=400)
        user_query = st.chat_input("Chat with your documents", disabled=st.session_state.disable_rag_chat_input)

        handle_input(user_query, container=container)

        with st.sidebar:
            api_key = st.text_input(label="API key", type="password", placeholder="Insert API key", disabled=st.session_state.disable_go_button)

            if st.button("Submit", disabled=st.session_state.disable_go_button):
                with st.spinner("Inserting API key"):
                    if api_key:
                        Data_Store.pass_api_key_to_llm(key=api_key)

            docs = st.file_uploader("Upload your files, to chat with them", accept_multiple_files=True, disabled=st.session_state.disable_rag_chat_input)

            if st.button("Process", on_click=rag_vars, disabled=st.session_state.disable_rag_chat_input):
                with st.spinner("Processing"):
                    if docs:
                        run_in_background(task=add_to_rag_pipeline_in_background,
                                        kwargs={'vectordb':st.session_state.vectordb,
                                                'docs':docs,
                                                'output':st.session_state.rag_queue})

if st.session_state.is_go_button_clicked:
    get_video_result_if_ready()

if st.session_state.activate_error_dialogue:
    error_dialogue(st.session_state.error_type)
    st.session_state.activate_error_dialogue = False

if "result" in st.session_state and not st.session_state.is_error:
    get_quiz_question_if_ready()
    get_rag_pipeline_if_ready()

# TODO add error type for try block code in go button------Done
# ensure that error message for each section: video, rag, quizzess, diaplsy consistentnly---Done
# confirm if empty transcript data causes error for rag
# test empty quiz_question to see if it throws an error
# feature to insert api key in learn environment-----Done
# spinner when inserting api key in homepage----Done
# disable process button accordingly----Done
# in homepage after inserting api key if a user has already typed a search query when the submit button is clicked, the user query should be automatically trigered--Done