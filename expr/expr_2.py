"""
ensemble of expert

"""
import os 
import sys 
sys.path.append(os.getcwd())
sys.path.append('/root/hole_agent/multi-agent/')
import gzip
import os
from collections import defaultdict
from typing import Dict, Any, Callable

import evaluate
import rich
from langchain.chains.llm import LLMChain
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    AIMessagePromptTemplate
from langchain_openai import ChatOpenAI
from prompts.expr2.prompts import *
from entity.entitys import LLMOutput
from tools.tools import fetch_context
import uuid
from rich.console import Console
from rich.markdown import Markdown


from utils.func import union_group, save_history, gemma_parser, convert_message, retry

# from utils.func import union_group, ConvertSystemMessageToHuman, convert_message, gemma_parser, save_history, retry

METRIC_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

__config_list__ = [
    # {"model_name": "Qwen1.5-72b-chat-gptq-int4", "base_url": "http://113.31.110.212:9987/v1", "temperature": 0.3, },
    {"model_name": "qwen2.5:72b", "base_url": "http://localhost:11434/v1", "temperature": 0.7,
     "format": "openai"},
    # {"model_name": "gemma-7b", "base_url": "http://113.31.110.212:11003/v1", "temperature": 1, "format": "google"},
    # {"model_name": "gemma-7b", "base_url": "http://113.31.110.212:11003/v1", "temperature": 1, "format": "google"},
    {"model_name": "qwen2.5:72b", "base_url": "http://localhost:11434/v1", "temperature": 0.7,
     "format": "openai"},    
    {"model_name": "qwen2.5:72b", "base_url": "http://localhost:11434/v1", "temperature": 1.0,
     "format": "openai"},    # {"model_name": "vicuna-13b-v1.5-16k", "base_url": "http://113.31.110.212:7001/v1", "temperature": 0.7, },
    # {"model_name": "vicuna-13b-v1.5-16k", "base_url": "http://113.31.110.212:7001/v1", "temperature": 1,
    #  "format": "openai"},
    {"model_name": "qwen2.5:72b", "base_url": "http://localhost:11434/v1", "temperature": 1.0,
     "format": "openai"}
]


def knn_group(group: list[str]):
    result = {}
    for idx, target in enumerate(group):
        cx1 = len(gzip.compress(target.encode()))
        distance_from_x1 = []
        for j in range(0, len(group)):
            if idx == j:
                continue
            item = group[j]
            cx2 = len(gzip.compress(item.encode()))
            x1x2 = " \n [SEQ] \n ".join([target, item])
            cx1x2 = len(gzip.compress(x1x2.encode()))
            ncd = (cx1x2 - min(cx1, cx2)) / max(cx1, cx2)
            distance_from_x1.append({"idx": j, "distance": ncd})
        if len(distance_from_x1) > 0:
            sorted_dis = sorted(distance_from_x1, key=lambda x: x["distance"])
            result[idx] = sorted_dis[0]['idx']
    return union_group(result)


def bleu_group(group: list[str]):
    result = {}
    bleu_metric = evaluate.load(f"{METRIC_PATH}/metric/bleu/bleu.py")

    for i, target in enumerate(group):
        dis = []
        for j in range(0, len(group)):
            if i == j: continue
            res = bleu_metric.compute(predictions=[target], references=[group[j]])
            dis.append({"idx": j, "bleu": res['bleu']})

        if len(dis) > 0:
            sorted_dis = sorted(dis, key=lambda x: x["bleu"])
            result[i] = sorted_dis[0]['idx']
    return union_group(result)


class CooperateExp:
    def __init__(self, chain_list: list[LLMChain], history: dict, response: list):
        self.chain_list = chain_list
        self.history = history
        self.chief: LLMChain = LLMChain(prompt=SummaryTemplate ,llm=ChatOpenAI(model_name="qwen2.5:72b",
                                                             base_url="http://localhost:11434/v1",
                                                             openai_api_key="aaa", temperature=1))
        answer_list = [item['ans'] for item in response]
        self.v1 = self.chief.invoke({"answer_list": answer_list})

    def revise(self, round: int = 3, context: str = '') -> AIMessage:
        update = False
        for r in range(round):
            for chain in self.chain_list:
                name = chain.name
                message = self.history[name] + [
                    HumanMessagePromptTemplate.from_template(ReviseTemplate).format(answer=self.v1['text'],
                                                                                    context=context)]
                res = chain.llm.invoke(message)
                revised = parse_revise_result(res)
                if isinstance(revised, AIMessage):
                    self.v1 = {'text':revised.content}
                    self.history[name] = self.history[name] + [revised]
                    update = True
            if not update:
                break

        return self.v1

    @property
    def get_summary(self):
        if isinstance(self.v1,AIMessage):
            return self.v1
        elif isinstance(self.v1,dict):
            if 'text' in self.v1:
                return AIMessage(content=self.v1['text'])
            raise ValueError("no text in v1 dictionary")
        else:
            raise ValueError("Unexpected type for get_summary")
    @property
    def chat_history(self):
        return self.history

    @property
    def chains(self):
        return self.chain_list


class DebateExp:

    def __init__(self,u1:CooperateExp,u2:CooperateExp):
        self.debate_teacher = ChatOpenAI(model_name="qwen2.5:72b",
                                                             base_url="http://localhost:11434/v1",
                                                             openai_api_key="aaa", temperature=1)
        self.u1 = u1
        self.u2 = u2
        self.u1_dialogue = []
        self.u2_dialogue = []

    @staticmethod
    @retry(tries=3)
    def make_conversation(llm,dialogue,parsar:Callable):
        result = llm.invoke(dialogue)
        return parsar(result)

    def debate_of_two(self,query:str,round: int = 3,context: str = '' )->AIMessage:
        message = [HumanMessagePromptTemplate.from_template(DebateTemplate).
                   format(yours=self.u2.get_summary.content, mine=self.u1.get_summary.content,
                          context='', query=query)]
        self.u1_dialogue= message
        self.make_conversation(llm=self.u1.chief.llm,dialogue=self.u1_dialogue,parsar=parse_debate_json)
        _reviews = self.u1.chief.llm.invoke(self.u1_dialogue)
        reviews = _reviews #parse_debate_json(_reviews)
        sw_1:STRENGTH_WEAK = parse_debate_json(_reviews)
        self.u1_dialogue = self.u1_dialogue + [reviews]
        self.u2_dialogue = [HumanMessagePromptTemplate.from_template(DebateTemplate).
                   format(mine=self.u2.get_summary.content,yours=self.u1.get_summary.content,
                          context='',query=query)]
        rebuttal = self.u2.chief.llm.invoke(self.u2_dialogue)
        sw_2:STRENGTH_WEAK = parse_debate_json(rebuttal)
        self.u2_dialogue.append(rebuttal)
        self.u1_dialogue = [HumanMessagePromptTemplate.from_template(DebateTemplate).
                   format(mine=self.u2.get_summary.content,yours=self.u1.get_summary.content,
                          context='',query=query)] + self.reverse_role(self.u2_dialogue[1:])
        final_answer = self.debate_teacher.invoke(ChatPromptTemplate.from_template(DEBATE_REVISE_TEMPLATE)
                                   .format(query=query,answer_1=self.u1.get_summary,answer_2=self.u2.get_summary,
                                           strength1=sw_1['MY_STRENGTH'],weakness1=sw_1['YOUR_WEAKNESS'],
                                           strength2=sw_2['MY_STRENGTH'],weakness2=sw_2['YOUR_WEAKNESS'],
                                           context=''
                                           ))
        return parse_with_keyword(final_answer,'FINAL_ANSWER')


    @staticmethod
    def reverse_role(message_list:List[BaseMessage]):
        out = []
        for message in message_list:
            if isinstance(message,HumanMessage):
                out.append(AIMessage(content=message.content))
            elif isinstance(message,AIMessage):
                out.append(HumanMessage(content=message.content))
            elif isinstance(message,SystemMessage):
                out.append(SystemMessage(content=message.content))
        return out







class DomainBase(object):

    def __init__(self, num_expert: int, expert_config_list: list[dict], system_prompt: str,
                 human_prompt: str, domain: str):
        assert len(
            expert_config_list) == num_expert, f"len(expert_config_list)={len(expert_config_list)} != {num_expert}"
        self.expert_chain = []
        self.chat_history = defaultdict(list)
        self.domain = domain
        self.prompt = ChatPromptTemplate.from_messages([SystemMessage(content=system_prompt),
                                                        HumanMessagePromptTemplate.from_template(human_prompt)])
        for expert_config in expert_config_list:
            if expert_config['format'] == 'openai':
                chain = LLMChain(llm=ChatOpenAI(model_name=expert_config['model_name'],
                                                base_url=expert_config['base_url'],
                                                temperature=expert_config['temperature'],
                                                openai_api_key="abc"), prompt=self.prompt,
                                 output_parser=gemma_parser())
            elif expert_config['format'] == 'google':
                chain = LLMChain(llm=ChatOpenAI(model_name=expert_config['model_name'],
                                                base_url=expert_config['base_url'],
                                                temperature=expert_config['temperature'],
                                                openai_api_key="abc"), prompt=convert_message(self.prompt, "google"),
                                 output_parser=gemma_parser()
                                 )
            else:
                raise NotImplementedError(expert_config['format'])
            setattr(chain, 'name', uuid.uuid4().hex + "_" + expert_config['model_name'] + "_" + expert_config['format'])
            self.expert_chain.append(chain)

    def initial_chat(self, query: str, enable_fetch: bool) -> list[dict[str, str]]:
        context = "无"
        if enable_fetch:
            context = fetch_context(query, mode="large", top_k=2)
        response = []
        for expert in self.expert_chain:
            rich.print(f"[green] fetching {expert.name} [/green]")
            resp: LLMOutput = expert.invoke({"question": query, "context": context})
            response.append({"expert_id": expert.name, "ans": resp['text']})
            self.chat_history[expert.name] = save_history(expert, resp)

        return response

    @staticmethod
    def bleu_similarity_calculate(answer_list: list[dict[str, str]]) -> dict[str, float]:
        bleu = evaluate.load(f"{METRIC_PATH}/metric/bleu/bleu.py")
        score = {}
        for idx, expert_resp in enumerate(answer_list):
            expert = expert_resp["expert_id"]
            pre_answer = expert_resp["ans"]
            reference = [item['ans'] for item in answer_list[0:idx] + answer_list[idx + 1:]]
            result = bleu.compute(predictions=[pre_answer], references=[reference])
            score[expert] = result['bleu']
        return score

    def divide_group(self, response: list[dict[str, str]], mode: str = 'knn') -> list[int]:
        if mode == 'knn':
            knn = knn_group([item['ans'] for item in response])
            return knn
        if mode == 'bleu':
            blue = bleu_group([item['ans'] for item in response])
            return blue

    @property
    def chains(self):
        return self.expert_chain

    @property
    def history(self):
        return self.chat_history


class Exp2:
    def __init__(self, domains: list[DomainBase], revise_round: int = 3):
        self.domains = domains
        self.revise_round = revise_round

    def get_cooperate_crack(self, query: str, enable_fetch: bool, mode: str = 'bleu') -> list[dict[str, str]]:
        if mode == 'bleu':
            domain_resp_list = []
            domain_answers = []
            for domain in self.domains:
                domain_response = domain.initial_chat(query, enable_fetch)
                domain_resp_list.append(domain_response)
                divide_group = domain.divide_group(domain_response, mode)
                cooperate_exp_list = []
                if len(divide_group) >= 1:
                    score = list(domain.bleu_similarity_calculate(domain_response).items())
                    score = sorted(score, key=lambda x: x[1], reverse=True)
                    ans = [item['ans'] for item in domain_response if item['expert_id'] == score[0][0]][0]
                    domain_answers.append({domain.domain: ans})
                else:
                    rich.print("grouping and debate==>")
                    for grp in divide_group:
                        group_chains: List[LLMChain] = [expert for idx, expert in enumerate(domain.expert_chain) if
                                                        idx in grp]
                        group_response = [response for idx, response in enumerate(domain_response) if idx in grp]
                        group_chain_names = [expert.name for expert in group_chains]
                        group_history = {k: v for k, v in domain.history.items() if k in group_chain_names}
                        exp = CooperateExp(chain_list=group_chains, history=group_history, response=group_response)
                        cooperate_exp_list.append(exp)
                        _ = exp.revise(self.revise_round)
                        grp_final = exp.get_summary
                        domain_answers.append({domain.domain: grp_final})
                    else:
                        debate_exp = DebateExp(cooperate_exp_list[0],cooperate_exp_list[1])
                        final_answer = debate_exp.debate_of_two(query=query,round=3)
                        domain_answers.append({domain.domain: final_answer.content})

            return domain_answers





if __name__ == '__main__':
    codeDomain = DomainBase(num_expert=4, expert_config_list=__config_list__,
                            system_prompt="你是一个python专家，请你帮助我解决python代码问题",
                            human_prompt=""" 现在我的问题是 {question} 可能用到的上下文信息 \n \n {context} """,
                            domain='code')

    exp2 = Exp2([codeDomain], revise_round=3)

    print(exp2.get_cooperate_crack(query="如何写一个带有参数的装饰器", enable_fetch=False, mode='bleu'))
