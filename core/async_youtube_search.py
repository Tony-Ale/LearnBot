import requests
import urllib.parse
import json
import aiohttp
import asyncio
from collections.abc import Iterable


class YoutubeSearch:
    def __init__(self, search_terms:str|Iterable[str], max_results=None, num_workers=1):
        self.search_terms = search_terms
        self.max_results = max_results

        if isinstance(search_terms, Iterable) and not isinstance(search_terms, str):
            self.videos = asyncio.run(self._multiple_async_search(num_workers))
        else:
            self.videos = self._search()

    def _search(self):
        encoded_search = urllib.parse.quote_plus(self.search_terms)
        BASE_URL = "https://youtube.com"
        url = f"{BASE_URL}/results?search_query={encoded_search}"
        response = requests.get(url).text
        while "ytInitialData" not in response:
            response = requests.get(url).text
        results = self._parse_html(response)
        if self.max_results is not None and len(results) > self.max_results:
            return results[: self.max_results]
        return results
    
    async def _async_search(self, query:str|list):
        async with aiohttp.ClientSession() as session:
            encoded_search = urllib.parse.quote_plus(query)
            BASE_URL = "https://youtube.com"
            url = f"{BASE_URL}/results?search_query={encoded_search}"
            result = await session.get(url)
            response = await result.text()
            while "ytInitialData" not in response:
                response = await result.text()
            results = self._parse_html(response)
            if self.max_results is not None and len(results) > self.max_results:
                return results[: self.max_results]
            return results
    
    async def _worker(self, query_queue:asyncio.Queue, response_queue:asyncio.Queue):
        while True:
            response = []
            
            # get query
            query = await query_queue.get()

            # perform task 
            try:
                response = await self._async_search(query)
            except Exception as e:
                for task in asyncio.all_tasks():
                    task.cancel(msg=f'Tasks were cancelled due to:\n{e}')

            # store response 
            response_queue.put_nowait({query:response})

            # Notify the queue that the query has been searched
            query_queue.task_done()

    async def _multiple_async_search(self, num_workers:int):
        courutine:list[asyncio.Task] = []

        query_queue = asyncio.Queue()
        response_queue = asyncio.Queue()

        # putting queries in queue
        for query in self.search_terms:
            query_queue.put_nowait(query)

        # assigning jobs 
        for _ in range(num_workers):
            task = asyncio.create_task(self._worker(query_queue, response_queue))
            courutine.append(task)

        await query_queue.join()

        # cancel the worker tasks
        for task in courutine:
            task.cancel()
        
        await asyncio.gather(*courutine, return_exceptions=True)

        # Retrieve results from the response queue
        results:dict[str, list] = {}
        while not response_queue.empty():
            result = await response_queue.get()
            results = results | result 
        
        return results


    def _parse_html(self, response):
        results = []
        start = (
            response.index("ytInitialData")
            + len("ytInitialData")
            + 3
        )
        end = response.index("};", start) + 1
        json_str = response[start:end]
        data = json.loads(json_str)

        for contents in data["contents"]["twoColumnSearchResultsRenderer"]["primaryContents"]["sectionListRenderer"]["contents"]:
            for video in contents["itemSectionRenderer"]["contents"]:
                res = {}
                if "videoRenderer" in video.keys():
                    video_data = video.get("videoRenderer", {})
                    res["id"] = video_data.get("videoId", None)
                    res["thumbnails"] = [thumb.get("url", None) for thumb in video_data.get("thumbnail", {}).get("thumbnails", [{}]) ]
                    res["title"] = video_data.get("title", {}).get("runs", [[{}]])[0].get("text", None)
                    res["long_desc"] = video_data.get("descriptionSnippet", {}).get("runs", [{}])[0].get("text", None)
                    res["channel"] = video_data.get("longBylineText", {}).get("runs", [[{}]])[0].get("text", None)
                    res["duration"] = video_data.get("lengthText", {}).get("simpleText", 0)
                    res["views"] = video_data.get("viewCountText", {}).get("simpleText", 0)
                    res["publish_time"] = video_data.get("publishedTimeText", {}).get("simpleText", 0)
                    res["url_suffix"] = video_data.get("navigationEndpoint", {}).get("commandMetadata", {}).get("webCommandMetadata", {}).get("url", None)
                    results.append(res)

            if results:
                return results
        return results

    def to_dict(self, clear_cache=True):
        result = self.videos
        if clear_cache:
            self.videos = ""
        return result

    def to_json(self, clear_cache=True):
        if isinstance(self.search_terms, Iterable):
            input_dict = {query:video for query, video in zip(self.search_terms, self.videos)}
            result = json.dumps(input_dict)
        else:
            result = json.dumps({"videos": self.videos})

        if clear_cache:
            self.videos = ""
        return result
