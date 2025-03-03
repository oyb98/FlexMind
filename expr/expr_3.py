from langchain_openai import ChatOpenAI

from expr_2 import DomainBase, DebateExp, Exp2
# from agents.agents import run
# from holo_gd.convert import dataloader,testdata
import json 
import sys 
import os 
sys.path.append(os.getcwd())
sys.path.append('/root/hole_agent/multi-agent/')
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

from prompts.prompt import MemoryAgentPrompt, FINAL_SUMMARY_TEMPLATE

from utils.configuration import MODEL, BASE_URL, API_KEY


class Expr3:
    def __init__(self, domainExpert: list[DomainBase], ensemble_mode: str = 'bleu'):
        self.domainExpert = domainExpert

        self.exp = Exp2(domainExpert, revise_round=3)
        self.ensemble_mode = ensemble_mode
        self.llm = ChatOpenAI(model_name=MODEL, base_url=BASE_URL, openai_api_key=API_KEY)
        self.input_prompt = FINAL_SUMMARY_TEMPLATE

    def run_diagnose(self, query: str, enable_fetch: bool, instance_id: str, timestamp: int):
        domain_answers = self.exp.get_cooperate_crack(query=query, enable_fetch=enable_fetch, mode=self.ensemble_mode)
        do = ""
        for domain_ans in domain_answers:
            domain, answer = domain_ans.keys(), domain_ans.values()
            do += f"{domain} exprt: \n result\n  {answer}\n\n"

        res = self.llm.invoke(self.input_prompt.format(domain=do, question=query))
        return res.content


if __name__ == '__main__':



    memoryDomain = DomainBase(num_expert=4, expert_config_list=__config_list__, system_prompt=MemoryAgentPrompt,
                              human_prompt=""" {question}. context message you may used \n \n {context} """,
                              domain="memory")

    exp3 = Expr3(domainExpert=[memoryDomain], ensemble_mode='bleu')
    query = """
    OOM error in the application
    """
    final_res = []
    res = exp3.run_diagnose(query=query, enable_fetch=False, instance_id="instance_id", timestamp=1713867180)
    print(res)

    
