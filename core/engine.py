from __future__ import annotations
from typing import TYPE_CHECKING
from .async_process_data import *
from .async_comment_downloader import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from async_youtube_transcript_api import YouTubeTranscriptApi, _errors
from .tuning_constants import *
from .data_store import Data_Store
import copy
import numpy as np
import random
import math
import asyncio
import time 

start = time.time()

def get_all_ids(query_id_map:dict[str, list[str]]):
    # get all ids 
    all_ids:list[str] = []
    for query_id in query_id_map.values():
        all_ids += query_id
    return all_ids

def process_all_ids_comments(query_id_map:dict[str, list[str]]):

    # get all ids 
    all_ids = get_all_ids(query_id_map)

    num_workers = min(Data_Store.max_workers, math.ceil(len(all_ids)/2))
    downloader = AsyncYoutubeCommentDownloader(num_workers=num_workers)
    id_comments_map = downloader.get_comments(all_ids)

    total_id_comments_prob_map = {}
    for video_id, comments in id_comments_map.items():

        if comments is None:
            comments = None
        else:
            comments = [comment['text'] for comment in comments]

        id_mean_map = sentiment_analysis(video_id, comments)

        total_id_comments_prob_map = total_id_comments_prob_map | id_mean_map

    return total_id_comments_prob_map


def get_sentiment_prob(comments:Iterable, sample_threshold=50):
    # interpretation of results gotten from this function:
    # The higher the return val, the more positive the sentiment is.
    if comments == None or len(comments) == 0:
        return NO_COMMENTS_PROB # punishing videos that dont have comments or comments are disabled.
    if len(comments) > sample_threshold:
        comments = random.sample(comments, k=sample_threshold)
    
    sentiment_analyzer = SentimentIntensityAnalyzer()
    pos_prob = 0
    for text in comments:
        result = sentiment_analyzer.polarity_scores(text)
        total = result['neg'] + result['pos']
        if total > 0:
            pos_prob += result['pos']/total
        else:
            lower_bound= 0
            upper_bound = result['neu']
            pos_val = random.uniform(lower_bound, upper_bound)
            pos_prob += pos_val/upper_bound
    mean = pos_prob/len(comments)
    return mean

def sentiment_analysis(video_id:str, comments:Iterable, sample_threshold=50):
    mean = get_sentiment_prob(comments, sample_threshold)

    return {video_id:mean}


"""Transcipt processing"""
def get_similarity_score_per_id(video_id:str, chunked_transcript:list[str], obj:TfidfVectorizer, llm_generated_desc:str):
    if chunked_transcript is None or obj is None:
        return {video_id:NO_TRANSCRIPT_PROB}
    
    id_similarity_score_map = similarity_analysis(video_id, obj, llm_generated_desc, chunked_transcript)

    return id_similarity_score_map

def get_all_chunked_transcripts(id_chunked_transcripts_map:dict[str, list[str] | None]):
    chunked_transcripts = []
    for chunked_transcript in id_chunked_transcripts_map.values():
        chunked_transcripts.append(chunked_transcript)
    return chunked_transcripts

def process_all_ids_transcript(query_id_map:dict[str, list[str]], all_llm_generated_desc:dict[str, str]):

    all_ids = get_all_ids(query_id_map)
    id_transcript_map = get_transcript(video_ids=all_ids)
    total_id_similarity_prob_map:dict[str, float] = {}

    for query, video_ids in query_id_map.items():
        id_chunked_transcripts_map = {}
        for video_id in video_ids:
            transcript = id_transcript_map[video_id]

            if transcript is None:
                id_chunked_transcripts_map[video_id] = transcript
            else:
                id_chunked_transcripts_map[video_id] = transcript_chunker(transcript)

        chunked_transcripts = get_all_chunked_transcripts(id_chunked_transcripts_map)
        documents = create_documents_from_chunked_transcript(chunked_transcripts)
        
        # for a case whereby no transcript was retrieved.
        try:
            obj = _fit_tfidf(documents)
        except:
            obj = None

        for video_id in video_ids:
            id_similarity_score_map = get_similarity_score_per_id(video_id=video_id,
                                                                    chunked_transcript=id_chunked_transcripts_map[video_id],
                                                                    obj=obj,
                                                                    llm_generated_desc=all_llm_generated_desc[query])
            
            total_id_similarity_prob_map = total_id_similarity_prob_map | id_similarity_score_map

    return total_id_similarity_prob_map

def create_documents_from_chunked_transcript(all_chunked_transcripts:list[list[str]]):
    documents = []
    for chunked_transcript in all_chunked_transcripts:
        if chunked_transcript is not None:
            documents += chunked_transcript
    
    return documents

def _fit_tfidf(documents):
    """documents: includes all transcripts and llm generated description"""
    vectorizer = TfidfVectorizer()  
    obj = vectorizer.fit(documents)
    return obj

def get_transcript_similarity_prob(obj:TfidfVectorizer, llm_generated_desc:str|list, transcript:str|list):

    if isinstance(llm_generated_desc, str):
        llm_generated_desc = [llm_generated_desc]
    if isinstance(transcript, str):
        transcript = [transcript]

    if len(llm_generated_desc) > 1:
        raise AssertionError("The list to be compared should have a length of one")
    
    expected_video_desc_score = obj.transform(llm_generated_desc)
    transcript_score = obj.transform(transcript)
    similarity_score = cosine_similarity(transcript_score, expected_video_desc_score)
    prob_score = _map_cosine_similarity_score_to_prob(similarity_score)
    
    return prob_score

def _map_cosine_similarity_score_to_prob(score:np.ndarray):
    """
    When s=1: The transformation yields p=1 (maximum similarity).
    When s=0: The transformation yields p=0.5 (neutral similarity).
    When s=-1: The transformation yields p=0 (maximum dissimilarity).
    """
    prob = (score + 1)/2
    prob = np.mean(prob)
    prob = normalize_prob(prob.item())

    return prob

def normalize_prob(prob:float):
    if prob <= 0.5:
        return MIN_NORMALIZED_COSINE_SCORE_PROB
    
    if prob > 0.5:
        return (prob-0.5)/0.5

def similarity_analysis(video_id:str, obj:TfidfVectorizer, llm_generated_desc:str|list, transcript:str|list):
    prob_score = get_transcript_similarity_prob(obj, llm_generated_desc, transcript)

    return {video_id:prob_score}

def process_transcripts(id_transcript_map:dict[str, list[str]|None], max_num_words:int):
    for video_id, transcript_data in id_transcript_map.items():
        transcript = []
        total_words_count = 0
        if transcript_data is not None:
            for text in transcript_data:
                transcript.append(text['text'])
                total_words_count += len(text['text'].split())

            average_words_per_line = total_words_count/len(transcript)
            # To control the number of words used for semantaic analysis and to reduce computational cost
            if  total_words_count > max_num_words:
                num_lines = math.floor(max_num_words/average_words_per_line) # to prevent going above sample size 
                transcript = random.sample(transcript, k=num_lines)
            id_transcript_map[video_id] = transcript
    return id_transcript_map

def get_transcript(video_ids:list[str], max_num_words=1000):
    # it checks if en is available. if it is not available, it checks for en-US
    num_workers = min(Data_Store.max_workers, math.ceil(len(video_ids)/2))
    output = YouTubeTranscriptApi.get_transcripts(video_ids, 
                                                    num_workers=num_workers, 
                                                    continue_after_error=True, 
                                                    languages=['en', 'en-US'])
    
    if Data_Store.activate_data_store:
        Data_Store.transcripts = copy.deepcopy(output[0])

    id_transcript_map, unretrieved_videos = output
    processed_id_transcript_map = process_transcripts(id_transcript_map, max_num_words=max_num_words)

    return processed_id_transcript_map

    
def transcript_chunker(transcript:list[str], chunk_num_words=100, max_num_words=1000):
    if transcript is None:
        return None
    
    num_lines = len(transcript)
    average_words_per_line = max_num_words/num_lines

    chunk_size  = math.floor(chunk_num_words/average_words_per_line)

    if chunk_size == 0:
        chunk_size = 1

    transcript_chunks = [transcript[i:i+chunk_size] for i in range(0, len(transcript), chunk_size)]

    chunks = [" ".join(chunk) for chunk in transcript_chunks]
    return chunks

#-------------------------------Video id time map analysis----------------------------------#
def process_id_time_map(query_result:dict[str, list[dict]], query_id_map:dict[str, list[str]]):

    id_time_map = video_id_time_map_generator(query_result)
    id_time_prob_map:dict[str, float] = {}

    for video_ids in query_id_map.values():
        prob_store = []
        for video_id in video_ids:
            prob_store.append(id_time_map[video_id])

        sum_val = sum(prob_store)

        # Normalize values 
        for video_id in video_ids:
            id_time_prob_map[video_id] = id_time_map[video_id]/sum_val

    return id_time_prob_map

#------------------------------------Total probability-------------------------------------#

def normalize_total_probabilities(query_id_prob_map:dict[str, dict[str, float]]):
    """Performs inplace update on query_id_prob_map"""
    for query, id_prob_map in query_id_prob_map.items():
        total_sum = sum(id_prob_map.values())
        for video_id, prob in id_prob_map.items():
            query_id_prob_map[query][video_id] = prob/total_sum
    return query_id_prob_map

def process_query_id_total_probability_map(query_id_map:dict[str, list[str]], 
                                           total_id_comments_prob_map:dict[str, float],
                                           total_id_similarity_prob_map:dict[str, float],
                                           id_time_prob_map:dict[str, float]):
    
    query_id_prob_map:dict[str, dict[str, float]] = {}

    for query, video_ids in query_id_map.items():
        for video_id in video_ids:
            if video_id in total_id_comments_prob_map:
                comment_prob = total_id_comments_prob_map[video_id]
            
            if video_id in total_id_similarity_prob_map:
                similarity_prob = total_id_similarity_prob_map[video_id]
            
            if video_id in id_time_prob_map:
                time_prob = id_time_prob_map[video_id]

            total_prob = comment_prob * similarity_prob #* time_prob

            if query not in query_id_prob_map:
                query_id_prob_map[query] = {}
            
            query_id_prob_map[query][video_id] = total_prob

    query_id_prob_map = normalize_total_probabilities(query_id_prob_map)
    return query_id_prob_map

def create_url(id):
    return f"https://www.youtube.com/watch?v={id}"

def get_top_result(query_id_prob_map:dict[str, dict[str, float]]):
    top_result_map:dict[str, tuple[str, float]] = {}
    for query, id_prob_map in query_id_prob_map.items():
        max_prob_id = max(id_prob_map, key=id_prob_map.get)
        watch_url = create_url(max_prob_id)

        # to ensure no duplicates in url
        for val in top_result_map.values():
            if watch_url in val:
                break
        else:
            """if watch url is not found in top_result_map, it adds the url to the map"""    
            top_result_map[query] = (watch_url, id_prob_map[max_prob_id])
    return top_result_map

def query_to_llm_generated_desc_map(queries:list, llm_generated_desc:list):
    all_llm_generated_desc = dict(zip(queries, llm_generated_desc))

    return all_llm_generated_desc

if __name__ == '__main__':
    random.seed(456)
    downloader = AsyncYoutubeCommentDownloader()
    #comments = downloader.get_comments_from_url('https://www.youtube.com/watch?v=ScMzIvxBSi4', sort_by=SORT_BY_POPULAR)
    

    queries = ['how to learn piano short videos' for _ in range(2)]
    queries = queries_validator(queries)
    result = get_query_result(queries)
    query_id_map = query_id_map_generator(queries, result)
    print(query_id_map)
    #query_id_map = {'how to learn piano short videos': ['jjdksPsU4DI', '4SXQ_wlbWog', '36oPnHzNzdI', 'Cs1Q6pd1Y7Y', 'dLlYHacPZNE', 'O2BeK6-PMos', 'wYNwSUblUSA', 'VcqS6gNStVg', 'fZ8X0RANX4c', 'CtwTknQYyyk', '1D3aWUx771A', '--LoWJPFGtA', 'vAw4wczTGJE', '9mgkN4gcrRI', 'cLOfXZ7W61s', 'ObIF8_vQmfY', 'p9tW3n6aO9Q']}
    id_comments_prob_map = process_all_ids_comments(query_id_map)

    all_llm_generated_desc = {'how to learn piano short videos':'piano vidoes, melody, harmony, beats music, chords, c, a, b, d'}
    id_similarity_prob_map = process_all_ids_transcript(query_id_map, all_llm_generated_desc)

    id_time_prob_map = process_id_time_map(result, query_id_map)

    total_prob_map = process_query_id_total_probability_map(query_id_map, id_comments_prob_map, id_similarity_prob_map, id_time_prob_map)
    
    end= time.time()
    print(total_prob_map, end-start)