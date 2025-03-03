from langchain_core.prompts import PromptTemplate

MemoryAgentPrompt = """
You are a memory analysis expert.  Please analyze the question I gave from the perspective of CPU.

You may or may not use context knowledge to answer questions.

Tell me what's the root cause for the given question from memory aspect. 
If this problem dont related to 

"""

CPUAgentPrompt = """
You are a CPU analysis expert. Please analyze the question I gave from the perspective of CPU.

You may or may not use context knowledge to answer questions.

Tell me what's the root cause for the given question from CPU aspect.

"""

SummaryAgentPrompt = """
You need to summarize and arrive at the final answer based on the questions I gave and the domain knowledge input summarized by other experts.

First of all, please tell me whether this problem is caused by incorrect operation by the user.
If not, please tell me what kind of problem the machine has.

Possible candidate rootCause include the following types:

1. Machine hardware failure
2. Version bug causes node coredump
3. Memory oom causes node failover
4. The number of connections is full
5. Business query or write volume increases, exceeding the instance capacity range
6. Storage cluster pressure increases
7. Other


Output format:

```
The issue is caused by [user/machine]

RootCause: \n
[RootCause, (give me the top3  possible rootcause in rootcause candidate list)] 

Proof: \n
[your proof, the reason why you think it is the rootcause,add other expert's analysis here if needed]

```

"""



TS_Prompt = """
You are an instance analysis expert who analyzes machine instance timeseries data. 
We provide you with an instance instance_id and timestamp. 

Here is the tool for you to use: {tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

 Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}

Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation'''




"""


TS_HUMAN = """

You need to analyze all the following types of instance metrics one by one.
['holo:hologres_master_dead_worker_count','holo:hologres_fe_connection_usage','holo:hologres_query_time','holo:hologres_instance_memory','holo:hologres_fe_query_failed_qps', 'holo:hologres_instance_cpu']

instance_id: {instance_id}

{agent_scratchpad}

 (reminder to respond in a JSON blob no matter what,and check if all metrics have been analyzed.)
"""


TS_SUMMARY_PROMPT = """
如下是holo发生异常时候的诊断知识，请根据这些知识回答我的问题

根因类型：机器硬件故障
人工分析思路：看hologram_master_dead_worker_count指标是否有徒增异常，同时机器自愈平台是否有相关节点维修记录。

根因类型：内存oom引起节点failover
人工分析思路：看hologram_master_dead_worker_count、worker_memory_used_bytes 两个指标同时异常增长。
具体实现思路：元仓

根因类型：连接数打满
人工分析思路：看hologram_fe_connection_usage 连接数使用率指标是否接近1.0,
具体实现思路：元仓中可以用关键词查到连接数打满的sql，根据这个sql定位到具体用户

根因类型：版本bug引起节点coredump
人工分析思路：看hologram_master_dead_worker_count指标是否有徒增异常，同时coredump 平台是否有相关节点记录。

根因类型：业务量查询或写入增大，超出实例容量范围
人工分析思路：看实时写入或者查询指标是否有徒增，同时cpu水位升高，query 报错指标变多。

根因类型：存储集群压力变大
人工分析思路：看实例存储写入或查询延时变大，同时query 报错指标变多。


我的问题是：
给定一些指标异常的信息，请你告诉我这个问题的根因是什么？
如下是相关的指标信息

{summary}

输出格式：

根因类型: [上述问题的根因]
证据：[XXX]

"""



FINAL_SUMMARY_TEMPLATE = """
现在有许多意见针对问题 {question}

{domain}

请你整合不同专家的结论给出最终的答案，答案需要使用如下的格式

输出格式：

问题定界：[机器问题/用户问题]两者中之一。 机器问题是指机器硬件问题导致的任务失败；用户问题是指用户的SQL语句不合理带来的问题
根因类型：根因需要是如下中的一个，如果是用户问题则无需给出根因
[1.机器硬件故障； 2.内存out of memory； 3.连接数(fe_connection)满；4.版本bug引起coredump； 5.业务量查询或写入增大超出实例容量范围；6.存储集群压力变大；7.其他原因]

原因：整合众多专家结论，给出详细原因

输出的例子如下：

问题定界：[平台问题]
根因类型：[1.业务量查询或写入增大超出实例容量范围 ]
原因：XXXX

问题定界：[用户问题]
根因类型：[无]
原因：XXXX

"""

RAG_SUMMARY = """
总结这个问题
{question}

输出：问题描述的关键词 比如操作类型，错误

以下是一个输出的例子

操作类型：SELECT；错误：out of memory.
"""