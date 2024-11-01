from langchain_cohere import ChatCohere, CohereEmbeddings
#from dotenv import load_dotenv

#load_dotenv()

class Data_Store:
    activate_data_store = False
    transcripts:dict[str, list[dict]] = None 

    #embed_model = CohereEmbeddings(model="embed-english-v3.0")
    #llm = ChatCohere()

    max_workers = 5

    @classmethod
    def pass_api_key_to_llm(cls, key=None):
        cls.embed_model = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=key)
        cls.llm = ChatCohere(cohere_api_key=key)