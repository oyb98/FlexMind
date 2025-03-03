import os
from collections import defaultdict
import evaluate
import rich
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from utils.configuration import MODEL,BASE_URL
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from tools.tools import fetch_context
import uuid
from rich.console import Console
from rich.markdown import Markdown

gemma = {"model_name": "gemma-7b", "base_url": "http://113.31.110.212:11003/v1", "temperature": 0.7}
# vicuna = {"model_name": "vicuna-13b-v1.5-16k", "base_url": "http://113.31.110.212:7001/v1", "temperature": 0.7}

messages = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("现在我的问题是 {question} 可能用到的上下文信息 \n \n {context}")
])
llm = (messages | ChatOpenAI(model_name=MODEL,
                 base_url=BASE_URL,
                 temperature=1,
                 openai_api_key="abc"))

print(llm.invoke({"question":"write quick sort","context":'none'}))
