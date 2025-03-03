import json
import re
from typing import List

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import BaseLLMOutputParser
from langchain_core.output_parsers.base import T
from langchain_core.outputs import Generation
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from typing_extensions import TypedDict
from string import punctuation
punctuation = punctuation + "\n"

ReviseTemplate = """
你需要根据你的知识尽力的让我的回答更正确，完美。

我的答案是

{answer}

这是可能用到的一些知识你可以依靠这些知识或者依靠你自己的能力: {context}

输出格式: 请按照以下格式输出  

FLAG: 如果你修改了我的答案请输出YES，如果没有输出NO
ANSWER: 输出回答

"""

DebateTemplate = """
这个是问题 {query}

这是你的答案

{yours}

这是我的答案

{mine}

如下是可能运用到的上下文信息

{context}

请你与我开始争辩谁的答案更好，你需要说明为什么你的答案更好

输出格式： 请按照如下JSON格式进行输出,请严格按照下面的输出格式，不要输出任何别的信息。

{{ 
"MY_STRENGTH": []
"YOUR_WEAKNESS": []
}}

"""


class STRENGTH_WEAK(TypedDict):
    MY_STRENGTH: str
    YOUR_WEAKNESS: str


def parse_debate(rebuttals: AIMessage) -> AIMessage:
    rebuttal = re.findall("OPINION(.*)", rebuttals.content)[0]
    return AIMessage(content=rebuttal)


def parse_debate_json(message: AIMessage) -> dict:
    return json.loads(message.content)


DEBATE_REVISE_TEMPLATE = """
现在问题是: {query}
如下有若干个回答和其对应STRENGTH和WEAKNESS

格式为：

```
回答：
{answer}

STRENGTH:
{strength}

WEAKNESS:
{weakness}

```

请你整合这些答案的，根据每个答案的STRENGTH和WEAKNESS进行相对应的修改，然后给出整合后的结果

{answer_1}

STRENGTH:
{strength1}

WEAKNESS:
{weakness2}

答案二
{answer_2}

STRENGTH:
{strength2}

WEAKNESS:
{weakness1}


请你整合这两个答案的，根据每个答案的STRENGTH和WEAKNESS进行相对应的修改，你应该保留每一个答案中的STRENGTH部分并且对WEAKNESS进行相应的修改
然后给出整合后的结果

这是可能用到的上下文信息
{context}

输出格式：请按照如下格式进行输出,请严格按照下面的输出格式，不要输出任何别的信息。

FINAL_ANSWER: XXXX

"""


def parse_revise_result(result: AIMessage) -> AIMessage | bool:
    response = result.content
    flag = re.search("FLAG(.*)", response).group()
    if "YES" in flag:
        idx_tuple = re.search('ANSWER', response).span()
        return AIMessage(content=response[idx_tuple[1]:].lstrip(punctuation))
    if "NO" in flag:
        return False


def parse_with_keyword(message: AIMessage,keyword:str) -> AIMessage:
    span = re.search(keyword,message.content).span()
    return AIMessage(content=message.content[span[1]:])


# class ReviseOutputParser(BaseLLMOutputParser):
#
#     def parse_result(self, result: List[Generation], *, partial: bool = False) -> T:
#         response = result[0].text
#         result = re.search('答案(. *)',response)
#         group = result.group()
#         return group


SummaryTemplate = ChatPromptTemplate.from_messages([HumanMessagePromptTemplate.from_template("""
请你整合如下几个回答的内容（请不要忽略细节，越详细越好）

{answer_list}

输出格式：

答案: []

"""
                                                                                             )])
