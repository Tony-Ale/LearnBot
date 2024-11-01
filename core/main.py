from .engine import *
from .query_LLM import invoke_llm
from .data_store import Data_Store

def get_videos(user_query:str, activate_data_store=True):
    if activate_data_store:
        Data_Store.activate_data_store = True

    youtube_queries, llm_generated_desc = invoke_llm(user_query)

    youtube_result = get_query_result(youtube_queries)

    all_generated_desc_map = query_to_llm_generated_desc_map(youtube_queries, llm_generated_desc)

    query_id_map = query_id_map_generator(youtube_queries, youtube_result)

    id_comments_prob_map = process_all_ids_comments(query_id_map)

    id_similarity_prob_map = process_all_ids_transcript(query_id_map, all_generated_desc_map)

    id_time_prob_map = process_id_time_map(youtube_result, query_id_map)

    query_id_prob_map = process_query_id_total_probability_map(query_id_map, id_comments_prob_map, id_similarity_prob_map, id_time_prob_map)

    top_result_map = get_top_result(query_id_prob_map)

    return top_result_map

if __name__ == '__main__':
    import time
    start = time.time()
    prob = get_videos('china')
    end = time.time()
    print(prob, end-start)
