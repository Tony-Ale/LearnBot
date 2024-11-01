import faiss
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from PyPDF2 import PdfReader
from .data_store import Data_Store
import asyncio
import time

def load_doc(doc):
    #loader_web = WebBaseLoader("https://en.wikipedia.org/wiki/Muhammadu_Buhari")
    #web_page = loader_web.load()
    #print(pages)
    loader_pdf = PyPDFLoader(doc)
    pages = loader_pdf.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 10
    )

    splits = text_splitter.split_documents(pages)
    return splits

def chunker(text:str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 400,
        chunk_overlap = 100
    )
    
    splits = text_splitter.split_text(text)
    return splits

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    splits = chunker(text)
    return splits

def gen_chromadb(splits):
    #persist_dir = "./docs/chroma/"
    vectordb = Chroma.from_texts(
        texts=splits,
        embedding=Data_Store.embed_model,
        #persist_directory=persist_dir
    )
    return vectordb

async def async_gen_faiss(splits:list[str], ids:list[str]=None):

    vectordb = await FAISS.afrom_texts(
            texts=splits,
            embedding=Data_Store.embed_model,
            ids=ids,
            )
    return vectordb

def gen_faiss(splits:list[str], ids:list[str]=None):
    vectordb = asyncio.run(async_gen_faiss(splits, ids=ids))
    return vectordb

async def add_text_to_gen_faiss(splits:list[str], vectordb:FAISS, ids:list[str]=None, batch=70):
    """embed batch chunks per minute in order not to raise an error for rate limited apis"""
    for i in range(0, len(splits), batch):
        chunks = splits[i:i+batch]

        if ids is None:
            chunk_ids = None
        else:
            chunk_ids = ids[i:i+batch]

        start = time.time()
        await vectordb.aadd_texts(texts=chunks,
                                    ids=chunk_ids)
        end = time.time()

        period = end-start
        if period < 60 and i+batch<len(splits):
            await asyncio.sleep(60-period)

def add_text_to_faiss_vectordb(splits:list[str], vectordb:FAISS, ids:list[str]=None):
    asyncio.run(add_text_to_gen_faiss(splits, vectordb, ids))

def delete_embedding_in_faiss_vectordb(vectordb:FAISS, ids:list[str]):
    delete_status = vectordb.delete(ids=ids)
    return delete_status

def history_retriever_chain(vectordb):

    llm = Data_Store.llm

    retriever = vectordb.as_retriever()

    # history aware retrieval
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"), 
        ("user", "Use the conversation above as context")
    ])

    # retrieves based on the history and context
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

#print(vectordb.as_retriever().get_relevant_documents("summarize the document"))
def conversational_chain(retriever_chain):
    llm = Data_Store.llm

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context: \n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"), 
        ("user", "{input}")
    ])


    # creating a document chain allows us to passing a list of documents as context to the LLM
    document_chain = create_stuff_documents_chain(llm, prompt)

    # to place the context automatically we have to use pass in the document_chain to a retrieval_chain
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

    return retrieval_chain

def get_llm_response(user_input, vectordb, chat_history):
    history_chain = history_retriever_chain(vectordb)
    conversation_chain = conversational_chain(history_chain)
    response = conversation_chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })

    return response['answer']