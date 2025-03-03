import functools
from collections import defaultdict
from typing import Dict, Any, Optional, List, Type
from uuid import UUID

from langchain.chains.llm import LLMChain
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ChatMessage, AIMessage
from langchain_core.output_parsers import BaseLLMOutputParser
from langchain_core.output_parsers.base import T
from langchain_core.outputs import Generation
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.prompts.chat import _StringImageMessagePromptTemplate

from entity.entitys import LLMOutput


def retry(tries=3):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            e = None
            for i in range(tries):
                try:
                    res = func(*args, **kwargs)
                    break
                except Exception as exc:
                    e = exc
                    print("Retrying ")
                    print(e)
                    pass
            else:
                print(e)
                return e
            return res

        return wrapper

    return deco


class UnionFindSet(object):
    def __init__(self, data_list):
        self.father_dict = {}
        self.size_dict = {}

        for node in data_list:
            self.father_dict[node] = node
            self.size_dict[node] = 1

    def find(self, node):
        father = self.father_dict[node]
        if node != father:
            if father != self.father_dict[father]:
                self.size_dict[father] -= 1
            father = self.find(father)
        self.father_dict[node] = father
        return father

    def is_same_set(self, node_a, node_b):
        return self.find(node_a) == self.find(node_b)

    def union(self, node_a, node_b):
        if node_a is None or node_b is None:
            return

        a_head = self.find(node_a)
        b_head = self.find(node_b)

        if (a_head != b_head):
            a_set_size = self.size_dict[a_head]
            b_set_size = self.size_dict[b_head]
            if (a_set_size >= b_set_size):
                self.father_dict[b_head] = a_head
                self.size_dict[a_head] = a_set_size + b_set_size
            else:
                self.father_dict[a_head] = b_head
                self.size_dict[b_head] = a_set_size + b_set_size


def union_group(d: dict):
    keys = list(d.keys())
    union_set = UnionFindSet(list(range(max(keys) + 1)))
    for src, dst in d.items():
        union_set.union(src, dst)
    res = defaultdict(list)
    for k in keys:
        father = union_set.find(k)
        res[father].append(k)
    return list(res.values())


class ConvertSystemMessageToHuman(BaseCallbackHandler):

    # def on_chain_start(
    #     self,
    #     serialized: Dict[str, Any],
    #     messages: List[List[BaseMessage]],prompts, **kwargs: Any
    # ) -> Any:
    #     print(messages)
    #     print(prompts)

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        prompts[0] = prompts[0].replace("System", "Human")


def convert_message(message: List[BaseMessage] | ChatPromptTemplate, api: str = 'openai') -> List | ChatPromptTemplate:
    if 'openai' in api:
        return message
    elif 'google' in api:
        out = []
        f = False
        if isinstance(message, ChatPromptTemplate):
            message = message.messages
            f = True
        prefix = ""
        for msg in message:
            if isinstance(msg, SystemMessage):
                prefix = msg.content + "\n\n\n"
            elif isinstance(msg, HumanMessage):
                out.append(HumanMessage(content=prefix + msg.content))
            elif isinstance(msg, AIMessage):
                out.append(AIMessage(content=msg.content))
            elif isinstance(msg, HumanMessagePromptTemplate):
                out.append(HumanMessagePromptTemplate.from_template(prefix + msg.prompt.template))
        if f:
            return ChatPromptTemplate.from_messages(out)

        return out
    else:
        raise NotImplementedError(api)


class gemma_parser(BaseLLMOutputParser):

    def __init__(self):
        super().__init__()

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> T:
        return result[0].text

    def aparse_result(
            self, result: List[Generation], *, partial: bool = False
    ) -> T:
        return result[0].text







def save_history(expert: LLMChain, message: LLMOutput):
    out = []
    prompts: list[HumanMessagePromptTemplate | BaseMessage] = expert.prompt.messages
    for msg in prompts:
        if isinstance(msg, BaseMessage):
            out.append(msg)
        elif isinstance(msg, HumanMessagePromptTemplate):
            input_keys = msg.input_variables
            res = {key: message.get(key, '') for key in input_keys}
            out.append(msg.format(**res))
    ai_message = message['text']
    out.append(AIMessage(content=ai_message))
    return out


if __name__ == '__main__':
    chat = ChatPromptTemplate.from_messages([SystemMessage(content="111"),
                                             HumanMessagePromptTemplate.from_template("222 {x}")])
    print(convert_message(chat, 'google'))
