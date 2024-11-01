import streamlit as st
import streamlit.components.v1 as components
import re
from core import filter_transcript, RAG
from .init_vars import *

def extract_youtube_video_id(url):
    """
    Extracts the video ID from a YouTube watch URL.

    Args:
        url (str): The YouTube URL in the format 'https://www.youtube.com/watch?v=VIDEO_ID'.

    Returns:
        str: The video ID if found, otherwise None.
    """
    # Regular expression to match the watch URL format
    pattern = r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})"
    
    # Search for the video ID in the URL
    match = re.search(pattern, url)
    
    # Return the video ID if found, else return None
    return match.group(1) if match else None

def user_answer_memory():
    pass

# Function to render the video
def render_video(index):
    st.write(st.session_state.video_title[index])
    st.video(st.session_state.url[index])

def prev_video():
    # Decrement the index but keep it within bounds
    if st.session_state.video_index > 0:
        st.session_state.video_index -= 1

def next_video():
    # Increment the index but keep it within bounds
    if st.session_state.video_index < len(st.session_state.url) - 1:
        st.session_state.video_index += 1

def prev_question():
    # Decrement the index but keep it within bounds
    if st.session_state.question_index > 0:
        st.session_state.question_index -= 1

def next_question():
    # Increment the index but keep it within bounds
    if st.session_state.question_index < len(st.session_state.quiz_questions) - 1:
        st.session_state.question_index += 1

def disable_go_button_vars():
    st.session_state.disable_go_button_count = 0
    st.session_state.disable_go_button = True
    st.session_state.is_go_button_clicked = True


def reset_vars():
    video_index_vars()

    st.session_state.question_index = 0

    st.session_state.video_ids = []

    url_vars()

    mcq_vars()

    disable_go_button_vars()

    rag_vars()

    video_result_vars()

    error_handling_vars()

def change_radio_button_font_size(label, size, margin_top, paddingBottom):
    components.html(
        f"""
        <script>
            var elems = window.parent.document.querySelectorAll('div[class*="stRadio"] p');
            var elem = Array.from(elems).find(x => x.innerText == "{label}");
            elem.style.fontSize = '{size}px'; // the fontsize you want to set it to
            elem.style.marginTop = '{margin_top}px';
            elem.style.paddingBottom = '{paddingBottom}px';
        </script>
        """
    )

def map_llm_generated_query_to_mcq(mcqs:dict[str, list[dict]], video_analysis_result:dict[str, tuple[str, float]]):
    """modifies mcq inplace"""
    for llm_generated_query, url_score_tuple in video_analysis_result.items():
        video_id = extract_youtube_video_id(url_score_tuple[0])
        if video_id in mcqs:
            for question_dict in mcqs[video_id]:
                question_dict['llm_generated_query'] = llm_generated_query
    return mcqs

def merge_questions_for_all_videos(mcqs:dict[str, list[dict]]):
    all_mcqs = []

    for mcq in mcqs.values():
        all_mcqs += mcq
    
    return all_mcqs


def handle_input(user_input, container, context_num=4):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [RAG.AIMessage(content="Hello, i am doc bot, how can i help you?")]

    if user_input:
        if len(st.session_state.chat_history)>context_num:
            # only use the last four (context_num=4) as history
            context = st.session_state.chat_history[-context_num:]
        else:
            context = st.session_state.chat_history

        response = st.session_state.conversation_chain.invoke({
            "chat_history": context,
            "input": user_input
        })

        st.session_state.chat_history.append(RAG.HumanMessage(content=user_input))
        st.session_state.chat_history.append(RAG.AIMessage(content=response['answer']))

    with container:
        if st.session_state.disable_rag_chat_input:
            if st.session_state.is_error:
                with st.chat_message("AI"):
                    st.write("An Error occurred while processing the videos or documents")
            else:
                with st.chat_message("AI"):
                    st.write("I am processing the videos or your document. This may take a few moments")
        else:
            for message in st.session_state.chat_history:
                if isinstance(message, RAG.AIMessage):
                    with st.chat_message("AI"):
                        st.write(message.content)
                elif isinstance(message, RAG.HumanMessage):
                    with st.chat_message("Human"):
                        st.write(message.content)

def prepare_transcripts_for_rag(transcripts:dict[str, list[dict]], results:dict[str, tuple[str, float]])->list[str]:
    extracted_transcripts = {}
    for _, (video_url, _) in results.items():
        id = extract_youtube_video_id(video_url)
        extracted_transcripts[id] =  transcripts[id]

    transcript_for_rag = []
    for transcript in extracted_transcripts.values():
        data = filter_transcript(transcript=transcript, max_num_words=1500)
        text = " ".join(data)
        chunked_texts = RAG.chunker(text)
        transcript_for_rag += chunked_texts
    return transcript_for_rag