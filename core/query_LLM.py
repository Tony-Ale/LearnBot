from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from .data_store import Data_Store
import json

def invoke_llm(user_query:str):
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="""You're an expert at crafting well-detailed learning outlines. Your task is to create a 5-step course outline based on a user's query. Your outline should be in JSON format, structured as follows:
                        WARNING: Note that your output should totally be in JSON FORMAT there should be no additional text, follow my instructions strictly.
                        WARNING: To create each subtopic title, first understand the user query, then identify the main keyword from the user's query. Then, make sure each title ends with 'in <main keyword of user query>'.
                        WARNING: "Content of Subtopics" should contain relevant text for performing similarity searches against video transcripts.
                        WARNING: If the users query is a sentence, understand the sentence, and then create the 5-step course outline as expected.
                    '{
                        "Step 1": {"Title of First Subtopic in <main keyword of user query>": "Content of First Subtopic - Describe in-depth and well detailed essay the key topics and areas that the user should cover for this subtopic"},
                        "Step 2": {"Title of Second Subtopic in <main keyword of user query>": "Content of Second Subtopic - Describe in-depth and well detailed essay the key topics and areas that the user should cover for this subtopic"},
                        "Step 3": {"Title of Third Subtopic in <main keyword of user query>": "Content of Third Subtopic - Describe in-depth and well detailed essay the key topics and areas that the user should cover for this subtopic"},
                        "Step 4": {"Title of Fourth Subtopic in <main keyword of user query>": "Content of Fourth Subtopic - Describe in-depth and well detailed essay the key topics and areas that the user should cover for this subtopic"},
                        "Step 5": {"Title of Fifth Subtopic in <main keyword of user query>": "Content of Fifth Subtopic - Describe in-depth and well detailed essay the key topics and areas that the user should cover for this subtopic"}
                    }'

                """),
        ("user", "using the format given to you, create a learning outline for the query given below\n\n{query}")
    ])


    llm = Data_Store.llm
    prompt = prompt_template.format_messages(query=user_query)
    output = llm.invoke(prompt)

    query_dict = output.content
    query_json = json.loads(query_dict)
    querys = []
    query_desc = []

    for key, val in query_json.items():
        for key2, val2 in val.items():
            querys.append(key2)
            query_desc.append(val2)
    
    return querys, query_desc