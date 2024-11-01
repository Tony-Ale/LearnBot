import http.cookiejar as cookiejar
CookieLoadError = (FileNotFoundError, cookiejar.LoadError)

from ._async_transcripts import TranscriptListFetcher

from ._errors import (
    CookiePathInvalid,
    CookiesInvalid,
    CouldNotRetrieveTranscript
)
import aiohttp
import asyncio
import traceback

class YouTubeTranscriptApi(object):
    session_created = False

    @classmethod
    async def async_list_transcripts(cls, video_id, proxies=None, cookies=None):
        """
        Retrieves the list of transcripts which are available for a given video. It returns a `TranscriptList` object
        which is iterable and provides methods to filter the list of transcripts for specific languages. While iterating
        over the `TranscriptList` the individual transcripts are represented by `Transcript` objects, which provide
        metadata and can either be fetched by calling `transcript.fetch()` or translated by calling
        `transcript.translate('en')`. Example::

            # retrieve the available transcripts
            transcript_list = YouTubeTranscriptApi.get('video_id')

            # iterate over all available transcripts
            for transcript in transcript_list:
                # the Transcript object provides metadata properties
                print(
                    transcript.video_id,
                    transcript.language,
                    transcript.language_code,
                    # whether it has been manually created or generated by YouTube
                    transcript.is_generated,
                    # a list of languages the transcript can be translated to
                    transcript.translation_languages,
                )

                # fetch the actual transcript data
                print(transcript.fetch())

                # translating the transcript will return another transcript object
                print(transcript.translate('en').fetch())

            # you can also directly filter for the language you are looking for, using the transcript list
            transcript = transcript_list.find_transcript(['de', 'en'])

            # or just filter for manually created transcripts
            transcript = transcript_list.find_manually_created_transcript(['de', 'en'])

            # or automatically generated ones
            transcript = transcript_list.find_generated_transcript(['de', 'en'])

        :param video_id: the youtube video id
        :type video_id: str
        :param proxies: a dictionary mapping of http and https proxies to be used for the network requests
        :type proxies: {'http': str, 'https': str} - http://docs.python-requests.org/en/master/user/advanced/#proxies
        :param cookies: a string of the path to a text file containing youtube authorization cookies
        :type cookies: str
        :return: the list of available transcripts
        :rtype TranscriptList:
        """
        if not cls.session_created:
            cls.session = aiohttp.ClientSession()
            cls.session_created = True

        if cookies:
            cls.session.cookie_jar = cls._load_cookies(cookies, video_id)
        # Deal with proxies in a different way
        transcript_list = await TranscriptListFetcher(cls.session).fetch(video_id)
        return transcript_list

    @classmethod
    def list_transcripts(cls, video_id, proxies=None, cookies=None):
        return asyncio.run(cls.async_list_transcripts(video_id, proxies, cookies))

    @classmethod
    def get_transcripts(cls, video_ids:list, num_workers:int, languages=('en',), continue_after_error=False, proxies=None,
                        cookies=None, preserve_formatting=False):
        
        transcripts = asyncio.run(cls._multiple_async_get_comments_from_url(video_ids, num_workers, continue_after_error, languages, proxies, cookies, preserve_formatting))
        return transcripts
    
    @classmethod
    def get_transcript(cls, video_id, languages=('en',), proxies=None, cookies=None, preserve_formatting=False):
        transcript = asyncio.run(cls.async_get_transcript(video_id, languages, proxies, cookies, preserve_formatting))
        asyncio.run(cls.session.close())
        return transcript
        
    @classmethod
    async def async_get_transcript(cls, video_id, languages=('en',), proxies=None, cookies=None, preserve_formatting=False):
        """
        Retrieves the transcript for a single video. This is just a shortcut for calling::

            YouTubeTranscriptApi.list_transcripts(video_id, proxies).find_transcript(languages).fetch()

        :param video_id: the youtube video id
        :type video_id: str
        :param languages: A list of language codes in a descending priority. For example, if this is set to ['de', 'en']
        it will first try to fetch the german transcript (de) and then fetch the english transcript (en) if it fails to
        do so.
        :type languages: list[str]
        :param proxies: a dictionary mapping of http and https proxies to be used for the network requests
        :type proxies: {'http': str, 'https': str} - http://docs.python-requests.org/en/master/user/advanced/#proxies
        :param cookies: a string of the path to a text file containing youtube authorization cookies
        :type cookies: str
        :param preserve_formatting: whether to keep select HTML text formatting
        :type preserve_formatting: bool
        :return: a list of dictionaries containing the 'text', 'start' and 'duration' keys
        :rtype [{'text': str, 'start': float, 'end': float}]:
        """
        assert isinstance(video_id, str), "`video_id` must be a string"
        transcript_list = await cls.async_list_transcripts(video_id, proxies, cookies)
        found_transcript = transcript_list.find_transcript(languages)
        return await found_transcript.fetch(preserve_formatting=preserve_formatting)
    
    @classmethod
    async def _worker(cls, 
                      id_queue:asyncio.Queue, 
                      response_queue:asyncio.Queue, 
                      unretrievable_videos_queue:asyncio.Queue, 
                      continue_after_error, 
                      *args, **kwargs):
        while True:

            # get query
            id = await id_queue.get()

            # perform task 
            try:
                response = await cls.async_get_transcript(id, *args, **kwargs)
            except Exception as e:
                unretrievable_videos_queue.put_nowait(id)
                if not continue_after_error:
                    for task in asyncio.all_tasks():
                        task.cancel(msg=f'Tasks were cancelled due to:\n{e}')
                    
                    await cls.session.close()
                    
                response_queue.put_nowait({id:None})
                id_queue.task_done()
                continue

            # store response 
            response_queue.put_nowait({id:response})

            # Notify the queue that the query has been searched
            id_queue.task_done()

    @classmethod
    async def _multiple_async_get_comments_from_url(cls, video_ids:list|str, num_workers:int, continue_after_error, *args, **kwargs):

        if isinstance(video_ids, str):
            video_ids = [video_ids]

        courutine:list[asyncio.Task] = []

        id_queue = asyncio.Queue()
        response_queue = asyncio.Queue()
        unretrievable_videos_queue = asyncio.Queue()

        # putting queries in queue
        for id in video_ids:
            id_queue.put_nowait(id)

        # assigning jobs 
        for _ in range(num_workers):
            task = asyncio.create_task(cls._worker(id_queue, 
                                                   response_queue, 
                                                   unretrievable_videos_queue, 
                                                   continue_after_error, 
                                                   *args, **kwargs))
            courutine.append(task)

        await id_queue.join()

        # cancel the worker tasks
        for task in courutine:
            task.cancel()
        
        await asyncio.gather(*courutine, return_exceptions=True)

        # Retrieve results from the response queue
        results:dict[str, list[dict]] = {}
        while not response_queue.empty():
            result = await response_queue.get()
            results = results | result

        # retrieve unretrievable videos id
        unretrievable_videos = []
        while not unretrievable_videos_queue.empty():
            video_id = await unretrievable_videos_queue.get()
            unretrievable_videos.append(video_id)

        
        # close session
        await cls.session.close()
        return results, unretrievable_videos

    @classmethod
    def _load_cookies(cls, cookies, video_id):
        try:
            cookie_jar = cookiejar.MozillaCookieJar()
            cookie_jar.load(cookies)
            if not cookie_jar:
                raise CookiesInvalid(video_id)
            return cookie_jar
        except CookieLoadError:
            raise CookiePathInvalid(video_id)