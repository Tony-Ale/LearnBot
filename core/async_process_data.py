from typing import Iterable
#from youtube_search import YoutubeSearch
from .async_youtube_search import YoutubeSearch
from .tuning_constants import DEFAULT_ID_TIME_PROB
import traceback
import asyncio

def get_query_result(queries:Iterable):
    num_workers = len(queries)
    query_to_query_result_map = YoutubeSearch(queries, max_results=3, num_workers=num_workers).to_dict()
    return query_to_query_result_map

def queries_validator(queries:list):
    """Ensures that there are no duplicate queries"""
    return set(queries)

def query_id_map_generator(queries:Iterable, query_results:dict[str, list[dict]]):

    query_id_map:dict[str, list] = {}
    for query in queries:
        each_query_result = query_results[query]
        for response_data in each_query_result:
            if query in query_id_map:
                query_id_map[query].append(response_data['id'])
            else:
                query_id_map[query] = [response_data['id']]
    return query_id_map

def video_id_time_map_generator(query_results:dict[str, list[dict]]):

    id_time_map:dict[str, float] = {}
    # Parse data
    for each_query_result in query_results.values():
        for response_data in each_query_result:
            video_id = response_data['id']
            if video_id not in id_time_map:
                try:
                    id_time_map[video_id] = view_string_to_int(response_data['views'])/time_string_to_years(response_data['publish_time'])
                except:
                    id_time_map[video_id] = DEFAULT_ID_TIME_PROB
                    traceback.print_exc()
                    print(response_data['views'], '\n\n', response_data['publish_time'])
    return id_time_map

def time_string_to_years(time_str:str):
    # Dictionary to store conversion factors to years
    time_to_years = {
        'second': 31536000,
        'minute': 525600,
        'hour': 1/8760,
        'day': 1/365,
        'week': 1/52,
        'month': 1/12,
        'year': 1
    }

    # Split the input string into number and time unit
    parts = time_str.split()
    number = int(parts[0])
    unit = parts[1].rstrip('s')  # Remove 's' from plural units (weeks, months, years)

    # Convert to years using the conversion dictionary
    if unit in time_to_years:
        return number * time_to_years[unit]
    else:
        raise ValueError(f"Unknown time unit: {unit}")
    
def view_string_to_int(view_str:str):
    parts = view_str.split()
    num_views_str = parts[0]
    parts_num_views_str = num_views_str.split(',')
    num_views = int("".join(parts_num_views_str))
    return num_views 

            
if __name__ == '__main__':
    import time

    start = time.time()
    queries = ['test' for _ in range(10)]

    result = get_query_result(queries)
    end = time.time()

    print(result, 'this is the time:', end-start)
    #query_id_map = query_id_map_generator(queries, result)
    #ideo_id_map = video_id_time_map_generator(result)
    #print(video_id_map)
    #print(len(result[0]))

