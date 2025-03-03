## DMA

Here we provide the code of DMA. Due to privacy issue, we didnt provide the knowledge base and the dataset we used.

We provide the demo code to demostrate the work flow of our multi agent framework.

## setup 

> pip install -r requirements.txt

and then replace all the api url and model to your own.

## DMA

please use `python expr/expr_3.py`to run our multi agent framework



```python

memoryDomain = DomainBase(num_expert=4, expert_config_list=__config_list__, system_prompt=MemoryAgentPrompt,
                            human_prompt=""" {question}. context message you may used \n \n {context} """,
                            domain="memory")

# you can add more domain of expert here
# e.g., cpuDomain 
# then add to the parameter `domainExpert` as 
exp3 = Expr3(domainExpert=[memoryDomain], ensemble_mode='bleu')

query = """
OOM error in the application
"""
final_res = []
# enable_fetch means whether use rag to enhance llm's answer
res = exp3.run_diagnose(query=query, enable_fetch=False, instance_id="instance_id", timestamp=1713867180)
print(res)

```