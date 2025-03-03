import os

import rich
from langchain.retrievers import ParentDocumentRetriever
from langchain.indexes import SQLRecordManager, index
from langchain.storage._lc_store import create_kv_docstore
from typing import Literal
from langchain.storage import InMemoryStore, LocalFileStore
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path
import requests
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from  langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from rich.progress import track
from langchain_community.vectorstores.chroma import Chroma
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys 
sys.path.append(os.getcwd())
sys.path.append('/root/hole_agent/multi-agent/')
from utils.configuration import MODEL, BASE_URL, API_KEY

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODE = Literal["small","large","full"]
def load_document():
    namespace = f"vector/holo_collection"
    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )
    record_manager.create_schema()
    loader = [ project_dir+"/knowledge/holo/" + item for item in os.listdir(project_dir + "/knowledge/holo") ]
    docs = []
    embeddings = HuggingFaceEmbeddings(model_name=project_dir + "/all-mpnet-base-v2")
    for loader in track(loader,description="Loading documents"):
        docs.extend(UnstructuredMarkdownLoader(loader,mode="single").load())
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    _is_vecstore_exist =False
    persistent_client = chromadb.PersistentClient(path=project_dir+"/chromadb")
    try:
        persistent_client.get_collection("holo_collection")
        rich.print("[green]loading holo collection vecstore[/green]")
        _is_vecstore_exist = True
    except Exception as e :
        rich.print("[red]holo_collection collection does not exist, will create one[/red]")

    vec_store = Chroma(client=persistent_client,collection_name="holo_collection", embedding_function=embeddings)

    fs = LocalFileStore(project_dir+"/holo_collection")
    store = create_kv_docstore(fs)
    retriever = ParentDocumentRetriever(
        vectorstore=vec_store,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    if not _is_vecstore_exist:
        retriever.add_documents(docs)
    else:
        retriever.docstore = store

    return retriever,vec_store,store



def fetch_context(question:str,mode:MODE='large',top_k:int=10,fetch_mode:str='api') -> list :
    """
    useful for fetching related document
    """
    if fetch_mode != 'api':
        rich.print("[green]Fetching....[/green]")
        model_name =  MODEL #os.environ.get("MODEL_NAME")
        base_url = BASE_URL # os.environ.get("BASE_URL")
        api_key = API_KEY
        prompt = ChatPromptTemplate.from_template("translate into english {question}. ")
        model = ChatOpenAI(model_name=model_name,base_url=base_url,openai_api_key=api_key)

        keyword = model.invoke(f"summary this {question}. ").content

        chain = (
            prompt | model | StrOutputParser()
        )
        rewrite_query = chain.invoke({"question":keyword})
        retriever,vec_store,store = load_document()
        sim_res:list[Document] = vec_store.similarity_search(keyword,k=top_k)
        if mode == "small":
            content = [ res.page_content for res in sim_res ]
        elif mode == 'large':
            ids = [ res.metadata['doc_id'] for res in sim_res]
            docs = store.mget(ids)
            content = [ res.page_content for res in docs ]
        elif mode == "full":
            raise NotImplementedError("full mode not implemented")
        rich.print(f"[green]fetched {content}....[/green]")

        return content
    else:

        payload = requests.get("http://127.0.0.1:4455/query?q="+question).json()
        content = payload['docs']
        return content 

        



if __name__ == '__main__':
    # kd_path = Path(__file__).parent.absolute() / 'knowledge' / 'hole'
    # load_document(kd_path)
    # retriever,vec_store,store = load_document()
    # out = vec_store.similarity_search("产生OOM的原因有哪些？")
    # print(out)
    content = fetch_context("insert sql out of memory","large",2)
    for item in content:
        print("=======================\n")
        rich.print(item)
