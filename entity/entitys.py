from typing import Type, Literal, List

from langchain_core.messages import BaseMessage,HumanMessage
from langchain_core.prompts.chat import _StringImageMessagePromptTemplate



from typing_extensions import TypedDict

LLMOutput = TypedDict('LLMOutput',{'question':str,'context':str,'text':str})