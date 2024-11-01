import threading 
import queue
import streamlit as st
from core import get_videos, Data_Store, generate_mcqs, RAG
from cohere.core.api_error import ApiError
from .errors import ErrorType
from .interface_funcs import map_llm_generated_query_to_mcq, merge_questions_for_all_videos
from .interface_funcs import prepare_transcripts_for_rag, extract_youtube_video_id

def handle_execeptions_in_other_threads(func:callable, **kwargs):
    try:
        func(**kwargs)
    except ApiError as e:
        if e.status_code == 401:
            # cohere api error
            kwargs['output'].put(ErrorType.API_KEY_ERROR)
        else:
            kwargs['output'].put(ErrorType.OTHER_ERRORS)
    except:
        kwargs['output'].put(ErrorType.OTHER_ERRORS)

def run_in_background(task:callable, kwargs:dict=dict(), *, handle_execeptions=True, exeception_func=handle_execeptions_in_other_threads):

    if handle_execeptions:
        kwargs['func'] = task
        task = exeception_func

    # Create and start a new thread
    thread = threading.Thread(target=task, kwargs=kwargs)
    thread.start()


def generate_mcqs_for_videos(video_ids:list,
                            map_answer_to_text:bool,
                            video_analysis_result:dict[str, tuple[str, float]],
                            output:queue.Queue,
                            *,
                            transcripts_data=None):
    
    """returns merged mcgs for all video_id"""
    mcqs = generate_mcqs(video_ids=video_ids, map_answer_to_text=map_answer_to_text, transcripts_data=transcripts_data)
    
    mcqs = map_llm_generated_query_to_mcq(mcqs, video_analysis_result)

    merged_mcqs = merge_questions_for_all_videos(mcqs)


    output.put(merged_mcqs)


def add_to_rag_pipeline_in_background(output:queue.Queue, **kwargs):
    pipeline_parts = add_to_rag_pipeline(**kwargs)
    output.put(pipeline_parts)

def add_to_rag_pipeline(vectordb:RAG.FAISS, docs:list=None, text_chunks:list[str]=None, ids:list[str]=None):
    if docs:
        text_chunks = RAG.get_pdf_text(docs)

    RAG.add_text_to_faiss_vectordb(splits=text_chunks, 
                                   vectordb=vectordb,
                                   ids=ids)

    history_chain = RAG.history_retriever_chain(vectordb)
    
    conversation_chain = RAG.conversational_chain(history_chain)

    return vectordb, conversation_chain

def get_videos_in_background(user_input, activate_data_store, output:queue.Queue):

    result = get_videos(user_query=user_input, activate_data_store=activate_data_store)

    text_chunks = prepare_transcripts_for_rag(Data_Store.transcripts, results=result)

    rag_ids = None#[str(uuid.uuid4()) for _ in range(len(text_chunks))]

    video_ids = []
    url = []
    video_title = []
    for llm_generated_query, (video_url, _) in result.items():
        video_id = extract_youtube_video_id(video_url)
        video_ids.append(video_id)

        url.append(video_url)
        video_title.append(llm_generated_query)

    thread_output = {"result":result,
                     "text_chunks":text_chunks,
                     "rag_ids":rag_ids,
                     "video_ids":video_ids,
                     "url":url,
                     "video_title":video_title}
    
    output.put(thread_output)

def get_mcq_and_prepare_rag_in_background():
    run_in_background(task=generate_mcqs_for_videos,  
                    kwargs={'video_ids':st.session_state.video_ids,
                            'map_answer_to_text':True,
                            'video_analysis_result':st.session_state.result,
                            'output':st.session_state.mcq_queue,
                            'transcripts_data': Data_Store.transcripts})
    
    run_in_background(task=add_to_rag_pipeline_in_background,
                    kwargs={'vectordb':st.session_state.vectordb,
                            'text_chunks':st.session_state.text_chunks,
                            'ids':st.session_state.rag_ids,
                            'output':st.session_state.rag_queue})
    

