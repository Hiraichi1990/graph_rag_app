import tempfile
import streamlit as st
from streamlit_chat import message

from langchain_core.runnables import (RunnableBranch, RunnableLambda, RunnableParallel,RunnablePassthrough)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field

from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import WikipediaLoader

from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader

from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import ConfigurableField, RunnableParallel, RunnablePassthrough

import getpass
import os

# 環境変数の読み込み
from dotenv import load_dotenv
load_dotenv()

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")
        
# (オプション)LangSmithを使う場合
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "Graph RAG"

# ドキュメントを取得
# raw_docs = WikipediaLoader(
#     query="ジョジョの奇妙な冒険",
#     lang="jp",
#     load_max_docs=300,
#     load_all_available_meta=False,
# ).load()
raw_docs = TextLoader('jojo.txt').load()

# テキストを分割
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)

# ドキュメントデータを分割
splitted_docs = text_splitter.split_documents(raw_docs[:3])

llm = ChatOpenAI(
        temperature=os.environ["OPENAI_API_TEMPERATURE"],
        model_name=os.environ["OPENAI_API_MODEL"],
        api_key=os.environ["OPENAI_API_KEY"],
        max_retries=5,
        timeout=60,
    )
llm_transformer = LLMGraphTransformer(llm=llm)
graph_docs = llm_transformer.convert_to_graph_documents(splitted_docs)

print(graph_docs)

