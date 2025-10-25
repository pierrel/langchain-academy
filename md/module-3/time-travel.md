[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-3/time-travel.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239536-lesson-5-time-travel)

# Time travel

## Review

We discussed motivations for human-in-the-loop:

(1) `Approval` - We can interrupt our agent, surface state to a user, and allow the user to accept an action

(2) `Debugging` - We can rewind the graph to reproduce or avoid issues

(3) `Editing` - You can modify the state 

We showed how breakpoints can stop the graph at specific nodes or allow the graph to dynamically interrupt itself.

Then we showed how to proceed with human approval or directly edit the graph state with human feedback.

## Goals

Now, let's show how LangGraph [supports debugging](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/) by viewing, re-playing, and even forking from past states. 

We call this `time travel`.


```python
%%capture --no-stderr
%pip install --quiet -U langgraph langchain_openai langgraph_sdk langgraph-prebuilt
```


```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
```

Let's build our agent.


```python
from langchain_openai import ChatOpenAI

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a by b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)
```


```python
from IPython.display import Image, display

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine the control flow
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
graph = builder.compile(checkpointer=MemorySaver())

# Show
display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
```


    
![jpeg](time-travel_files/time-travel_6_0.jpg)
    


Let's run it, as before.


```python
# Input
initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}

# Thread
thread = {"configurable": {"thread_id": "1"}}

# Run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Multiply 2 and 3
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      multiply (call_ikJxMpb777bKMYgmM3d9mYjW)
     Call ID: call_ikJxMpb777bKMYgmM3d9mYjW
      Args:
        a: 2
        b: 3
    =================================[1m Tool Message [0m=================================
    Name: multiply
    
    6
    ==================================[1m Ai Message [0m==================================
    
    The result of multiplying 2 and 3 is 6.


## Browsing History

We can use `get_state` to look at the **current** state of our graph, given the `thread_id`!


```python
graph.get_state({'configurable': {'thread_id': '1'}})
```




    StateSnapshot(values={'messages': [HumanMessage(content='Multiply 2 and 3', id='4ee8c440-0e4a-47d7-852f-06e2a6c4f84d'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_ikJxMpb777bKMYgmM3d9mYjW', 'function': {'arguments': '{"a":2,"b":3}', 'name': 'multiply'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 17, 'prompt_tokens': 131, 'total_tokens': 148}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-bc24d334-8013-4f85-826f-e1ed69c86df0-0', tool_calls=[{'name': 'multiply', 'args': {'a': 2, 'b': 3}, 'id': 'call_ikJxMpb777bKMYgmM3d9mYjW', 'type': 'tool_call'}], usage_metadata={'input_tokens': 131, 'output_tokens': 17, 'total_tokens': 148}), ToolMessage(content='6', name='multiply', id='1012611a-30c5-4732-b789-8c455580c7b4', tool_call_id='call_ikJxMpb777bKMYgmM3d9mYjW'), AIMessage(content='The result of multiplying 2 and 3 is 6.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 156, 'total_tokens': 170}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5', 'finish_reason': 'stop', 'logprobs': None}, id='run-b46f3fed-ca3b-4e09-83f4-77ea5071e9bf-0', usage_metadata={'input_tokens': 156, 'output_tokens': 14, 'total_tokens': 170})]}, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef6a440-ac9e-6024-8003-6fd8435c1d3b'}}, metadata={'source': 'loop', 'writes': {'assistant': {'messages': [AIMessage(content='The result of multiplying 2 and 3 is 6.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 156, 'total_tokens': 170}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5', 'finish_reason': 'stop', 'logprobs': None}, id='run-b46f3fed-ca3b-4e09-83f4-77ea5071e9bf-0', usage_metadata={'input_tokens': 156, 'output_tokens': 14, 'total_tokens': 170})]}}, 'step': 3, 'parents': {}}, created_at='2024-09-03T22:29:54.309727+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef6a440-a759-6d02-8002-f1da6393e1ab'}}, tasks=())



We can also browse the state history of our agent.

`get_state_history` lets us get the state at all prior steps.



```python
all_states = [s for s in graph.get_state_history(thread)]
```


```python
len(all_states)
```




    5



The first element is the current state, just as we got from `get_state`.


```python
all_states[-2]
```




    StateSnapshot(values={'messages': [HumanMessage(content='Multiply 2 and 3', id='4ee8c440-0e4a-47d7-852f-06e2a6c4f84d')]}, next=('assistant',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef6a440-a003-6c74-8000-8a2d82b0d126'}}, metadata={'source': 'loop', 'writes': None, 'step': 0, 'parents': {}}, created_at='2024-09-03T22:29:52.988265+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef6a440-9ffe-6512-bfff-9e6d8dc24bba'}}, tasks=(PregelTask(id='ca669906-0c4f-5165-840d-7a6a3fce9fb9', name='assistant', error=None, interrupts=(), state=None),))



Everything above we can visualize here: 

![fig1.jpg](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbb038211b544898570be3_time-travel1.png)

## Replaying 

We can re-run our agent from any of the prior steps.

![fig2.jpg](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbb038a0bd34b541c78fb8_time-travel2.png)

Let's look back at the step that recieved human input!


```python
to_replay = all_states[-2]
```


```python
to_replay
```




    StateSnapshot(values={'messages': [HumanMessage(content='Multiply 2 and 3', id='4ee8c440-0e4a-47d7-852f-06e2a6c4f84d')]}, next=('assistant',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef6a440-a003-6c74-8000-8a2d82b0d126'}}, metadata={'source': 'loop', 'writes': None, 'step': 0, 'parents': {}}, created_at='2024-09-03T22:29:52.988265+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef6a440-9ffe-6512-bfff-9e6d8dc24bba'}}, tasks=(PregelTask(id='ca669906-0c4f-5165-840d-7a6a3fce9fb9', name='assistant', error=None, interrupts=(), state=None),))



Look at the state.


```python
to_replay.values
```




    {'messages': [HumanMessage(content='Multiply 2 and 3', id='4ee8c440-0e4a-47d7-852f-06e2a6c4f84d')]}



We can see the next node to call.


```python
to_replay.next
```




    ('assistant',)



We also get the config, which tells us the `checkpoint_id` as well as the `thread_id`.


```python
to_replay.config
```




    {'configurable': {'thread_id': '1',
      'checkpoint_ns': '',
      'checkpoint_id': '1ef6a440-a003-6c74-8000-8a2d82b0d126'}}



To replay from here, we simply pass the config back to the agent!

The graph knows that this checkpoint has aleady been executed. 

It just re-plays from this checkpoint!


```python
for event in graph.stream(None, to_replay.config, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Multiply 2 and 3
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      multiply (call_SABfB57CnDkMu9HJeUE0mvJ9)
     Call ID: call_SABfB57CnDkMu9HJeUE0mvJ9
      Args:
        a: 2
        b: 3
    =================================[1m Tool Message [0m=================================
    Name: multiply
    
    6
    ==================================[1m Ai Message [0m==================================
    
    The result of multiplying 2 and 3 is 6.


Now, we can see our current state after the agent re-ran.

## Forking

What if we want to run from that same step, but with a different input.

This is forking.

![fig3.jpg](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbb038f89f2d847ee5c336_time-travel3.png)


```python
to_fork = all_states[-2]
to_fork.values["messages"]
```




    [HumanMessage(content='Multiply 2 and 3', id='4ee8c440-0e4a-47d7-852f-06e2a6c4f84d')]



Again, we have the config.


```python
to_fork.config
```




    {'configurable': {'thread_id': '1',
      'checkpoint_ns': '',
      'checkpoint_id': '1ef6a440-a003-6c74-8000-8a2d82b0d126'}}



Let's modify the state at this checkpoint.

We can just run `update_state` with the `checkpoint_id` supplied. 

Remember how our reducer on `messages` works: 

* It will append, unless we supply a message ID.
* We supply the message ID to overwrite the message, rather than appending to state!

So, to overwrite the the message, we just supply the message ID, which we have `to_fork.values["messages"].id`.


```python
fork_config = graph.update_state(
    to_fork.config,
    {"messages": [HumanMessage(content='Multiply 5 and 3', 
                               id=to_fork.values["messages"][0].id)]},
)
```


```python
fork_config
```




    {'configurable': {'thread_id': '1',
      'checkpoint_ns': '',
      'checkpoint_id': '1ef6a442-3661-62f6-8001-d3c01b96f98b'}}



This creates a new, forked checkpoint.
 
But, the metadata - e.g., where to go next - is perserved! 

We can see the current state of our agent has been updated with our fork.


```python
all_states = [state for state in graph.get_state_history(thread) ]
all_states[0].values["messages"]
```




    [HumanMessage(content='Multiply 5 and 3', id='4ee8c440-0e4a-47d7-852f-06e2a6c4f84d')]




```python
graph.get_state({'configurable': {'thread_id': '1'}})
```




    StateSnapshot(values={'messages': [HumanMessage(content='Multiply 5 and 3', id='4ee8c440-0e4a-47d7-852f-06e2a6c4f84d')]}, next=('assistant',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef6a442-3661-62f6-8001-d3c01b96f98b'}}, metadata={'source': 'update', 'step': 1, 'writes': {'__start__': {'messages': [HumanMessage(content='Multiply 5 and 3', id='4ee8c440-0e4a-47d7-852f-06e2a6c4f84d')]}}, 'parents': {}}, created_at='2024-09-03T22:30:35.598707+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef6a440-a003-6c74-8000-8a2d82b0d126'}}, tasks=(PregelTask(id='f8990132-a8d3-5ddd-8d9e-1efbfc220da1', name='assistant', error=None, interrupts=(), state=None),))



Now, when we stream, the graph knows this checkpoint has never been executed.

So, the graph runs, rather than simply re-playing.


```python
for event in graph.stream(None, fork_config, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Multiply 5 and 3
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      multiply (call_KP2CVNMMUKMJAQuFmamHB21r)
     Call ID: call_KP2CVNMMUKMJAQuFmamHB21r
      Args:
        a: 5
        b: 3
    =================================[1m Tool Message [0m=================================
    Name: multiply
    
    15
    ==================================[1m Ai Message [0m==================================
    
    The result of multiplying 5 and 3 is 15.


Now, we can see the current state is the end of our agent run.


```python
graph.get_state({'configurable': {'thread_id': '1'}})
```




    StateSnapshot(values={'messages': [HumanMessage(content='Multiply 5 and 3', id='4ee8c440-0e4a-47d7-852f-06e2a6c4f84d'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_KP2CVNMMUKMJAQuFmamHB21r', 'function': {'arguments': '{"a":5,"b":3}', 'name': 'multiply'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 17, 'prompt_tokens': 131, 'total_tokens': 148}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-bc420009-d1f6-49b8-bea7-dfc9fca7eb79-0', tool_calls=[{'name': 'multiply', 'args': {'a': 5, 'b': 3}, 'id': 'call_KP2CVNMMUKMJAQuFmamHB21r', 'type': 'tool_call'}], usage_metadata={'input_tokens': 131, 'output_tokens': 17, 'total_tokens': 148}), ToolMessage(content='15', name='multiply', id='9232e653-816d-471a-9002-9a1ecd453364', tool_call_id='call_KP2CVNMMUKMJAQuFmamHB21r'), AIMessage(content='The result of multiplying 5 and 3 is 15.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 156, 'total_tokens': 170}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5', 'finish_reason': 'stop', 'logprobs': None}, id='run-86c21888-d832-47c5-9e76-0aa2676116dc-0', usage_metadata={'input_tokens': 156, 'output_tokens': 14, 'total_tokens': 170})]}, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef6a442-a2e2-6e98-8004-4a0b75537950'}}, metadata={'source': 'loop', 'writes': {'assistant': {'messages': [AIMessage(content='The result of multiplying 5 and 3 is 15.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 156, 'total_tokens': 170}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5', 'finish_reason': 'stop', 'logprobs': None}, id='run-86c21888-d832-47c5-9e76-0aa2676116dc-0', usage_metadata={'input_tokens': 156, 'output_tokens': 14, 'total_tokens': 170})]}}, 'step': 4, 'parents': {}}, created_at='2024-09-03T22:30:46.976463+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef6a442-9db0-6056-8003-7304cab7bed8'}}, tasks=())



### Time travel with LangGraph API

**⚠️ DISCLAIMER**

Since the filming of these videos, we've updated Studio so that it can be run locally and opened in your browser. This is now the preferred way to run Studio (rather than using the Desktop App as shown in the video). See documentation [here](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/#local-development-server) on the local development server and [here](https://langchain-ai.github.io/langgraph/how-tos/local-studio/#run-the-development-server). To start the local development server, run the following command in your terminal in the `/studio` directory in this module:

```
langgraph dev
```

You should see the following output:
```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

Open your browser and navigate to the Studio UI: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`.

We connect to it via the SDK and show how the LangGraph API [supports time travel](https://langchain-ai.github.io/langgraph/cloud/how-tos/human_in_the_loop_time_travel/#initial-invocation). 


```python
if 'google.colab' in str(get_ipython()):
    raise Exception("Unfortunately LangGraph Studio is currently not supported on Google Colab")
```


```python
from langgraph_sdk import get_client
client = get_client(url="http://127.0.0.1:2024")
```

#### Re-playing 

Let's run our agent streaming `updates` to the state of the graph after each node is called.


```python
initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}
thread = await client.threads.create()
async for chunk in client.runs.stream(
    thread["thread_id"],
    assistant_id = "agent",
    input=initial_input,
    stream_mode="updates",
):
    if chunk.data:
        assisant_node = chunk.data.get('assistant', {}).get('messages', [])
        tool_node = chunk.data.get('tools', {}).get('messages', [])
        if assisant_node:
            print("-" * 20+"Assistant Node"+"-" * 20)
            print(assisant_node[-1])
        elif tool_node:
            print("-" * 20+"Tools Node"+"-" * 20)
            print(tool_node[-1])
```

    --------------------Assistant Node--------------------
    {'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': 'call_SG7XYqDENGq7mwXrnioNLosS', 'function': {'arguments': '{"a":2,"b":3}', 'name': 'multiply'}, 'type': 'function'}]}, 'response_metadata': {'finish_reason': 'tool_calls', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5'}, 'type': 'ai', 'name': None, 'id': 'run-2c120fc3-3c82-4599-b8ec-24fbee207cad', 'example': False, 'tool_calls': [{'name': 'multiply', 'args': {'a': 2, 'b': 3}, 'id': 'call_SG7XYqDENGq7mwXrnioNLosS', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': None}
    --------------------Tools Node--------------------
    {'content': '6', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'multiply', 'id': '3b40d091-58b2-4566-a84c-60af67206307', 'tool_call_id': 'call_SG7XYqDENGq7mwXrnioNLosS', 'artifact': None, 'status': 'success'}
    --------------------Assistant Node--------------------
    {'content': 'The result of multiplying 2 and 3 is 6.', 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_fde2829a40'}, 'type': 'ai', 'name': None, 'id': 'run-1272d9b0-a0aa-4ff7-8bad-fdffd27c5506', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}


Now, let's look at **replaying** from a specified checkpoint. 

We simply need to pass the `checkpoint_id`.


```python
states = await client.threads.get_history(thread['thread_id'])
to_replay = states[-2]
to_replay
```




    {'values': {'messages': [{'content': 'Multiply 2 and 3',
        'additional_kwargs': {'example': False,
         'additional_kwargs': {},
         'response_metadata': {}},
        'response_metadata': {},
        'type': 'human',
        'name': None,
        'id': 'df98147a-cb3d-4f1a-b7f7-1545c4b6f042',
        'example': False}]},
     'next': ['assistant'],
     'tasks': [{'id': 'e497456f-827a-5027-87bd-b0ccd54aa89a',
       'name': 'assistant',
       'error': None,
       'interrupts': [],
       'state': None}],
     'metadata': {'step': 0,
      'run_id': '1ef6a449-7fbc-6c90-8754-4e6b1b582790',
      'source': 'loop',
      'writes': None,
      'parents': {},
      'user_id': '',
      'graph_id': 'agent',
      'thread_id': '708e1d8f-f7c8-4093-9bb4-999c4237cb4a',
      'created_by': 'system',
      'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'},
     'created_at': '2024-09-03T22:33:51.380352+00:00',
     'checkpoint_id': '1ef6a449-817f-6b55-8000-07c18fbdf7c8',
     'parent_checkpoint_id': '1ef6a449-816c-6fd6-bfff-32a56dd2635f'}



Let's stream with `stream_mode="values"` to see the full state at every node as we replay. 


```python
async for chunk in client.runs.stream(
    thread["thread_id"],
    assistant_id="agent",
    input=None,
    stream_mode="values",
    checkpoint_id=to_replay['checkpoint_id']
):      
    print(f"Receiving new event of type: {chunk.event}...")
    print(chunk.data)
    print("\n\n")
```

    Receiving new event of type: metadata...
    {'run_id': '1ef6a44a-5806-6bb1-b2ee-92ecfda7f67d'}
    
    
    
    Receiving new event of type: values...
    {'messages': [{'content': 'Multiply 2 and 3', 'additional_kwargs': {'example': False, 'additional_kwargs': {}, 'response_metadata': {}}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'df98147a-cb3d-4f1a-b7f7-1545c4b6f042', 'example': False}]}
    
    
    
    Receiving new event of type: values...
    {'messages': [{'content': 'Multiply 2 and 3', 'additional_kwargs': {'example': False, 'additional_kwargs': {}, 'response_metadata': {}}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'df98147a-cb3d-4f1a-b7f7-1545c4b6f042', 'example': False}, {'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': 'call_Rn9YQ6iZyYtzrELBz7EfQcs0', 'function': {'arguments': '{"a":2,"b":3}', 'name': 'multiply'}, 'type': 'function'}]}, 'response_metadata': {'finish_reason': 'tool_calls', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5'}, 'type': 'ai', 'name': None, 'id': 'run-e60d82d7-7743-4f13-bebd-3616a88720a9', 'example': False, 'tool_calls': [{'name': 'multiply', 'args': {'a': 2, 'b': 3}, 'id': 'call_Rn9YQ6iZyYtzrELBz7EfQcs0', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': None}]}
    
    
    
    Receiving new event of type: values...
    {'messages': [{'content': 'Multiply 2 and 3', 'additional_kwargs': {'example': False, 'additional_kwargs': {}, 'response_metadata': {}}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'df98147a-cb3d-4f1a-b7f7-1545c4b6f042', 'example': False}, {'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': 'call_Rn9YQ6iZyYtzrELBz7EfQcs0', 'function': {'arguments': '{"a":2,"b":3}', 'name': 'multiply'}, 'type': 'function'}]}, 'response_metadata': {'finish_reason': 'tool_calls', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5'}, 'type': 'ai', 'name': None, 'id': 'run-e60d82d7-7743-4f13-bebd-3616a88720a9', 'example': False, 'tool_calls': [{'name': 'multiply', 'args': {'a': 2, 'b': 3}, 'id': 'call_Rn9YQ6iZyYtzrELBz7EfQcs0', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': '6', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'multiply', 'id': 'f1be0b83-4565-4aa2-9b9a-cd8874c6a2bc', 'tool_call_id': 'call_Rn9YQ6iZyYtzrELBz7EfQcs0', 'artifact': None, 'status': 'success'}]}
    
    
    
    Receiving new event of type: values...
    {'messages': [{'content': 'Multiply 2 and 3', 'additional_kwargs': {'example': False, 'additional_kwargs': {}, 'response_metadata': {}}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'df98147a-cb3d-4f1a-b7f7-1545c4b6f042', 'example': False}, {'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': 'call_Rn9YQ6iZyYtzrELBz7EfQcs0', 'function': {'arguments': '{"a":2,"b":3}', 'name': 'multiply'}, 'type': 'function'}]}, 'response_metadata': {'finish_reason': 'tool_calls', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5'}, 'type': 'ai', 'name': None, 'id': 'run-e60d82d7-7743-4f13-bebd-3616a88720a9', 'example': False, 'tool_calls': [{'name': 'multiply', 'args': {'a': 2, 'b': 3}, 'id': 'call_Rn9YQ6iZyYtzrELBz7EfQcs0', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': '6', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'multiply', 'id': 'f1be0b83-4565-4aa2-9b9a-cd8874c6a2bc', 'tool_call_id': 'call_Rn9YQ6iZyYtzrELBz7EfQcs0', 'artifact': None, 'status': 'success'}, {'content': 'The result of multiplying 2 and 3 is 6.', 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5'}, 'type': 'ai', 'name': None, 'id': 'run-55e5847a-d542-4977-84d7-24852e78b0a9', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}
    
    
    


We can all view this as streaming only `updates` to state made by the nodes that we reply.


```python
async for chunk in client.runs.stream(
    thread["thread_id"],
    assistant_id="agent",
    input=None,
    stream_mode="updates",
    checkpoint_id=to_replay['checkpoint_id']
):
    if chunk.data:
        assisant_node = chunk.data.get('assistant', {}).get('messages', [])
        tool_node = chunk.data.get('tools', {}).get('messages', [])
        if assisant_node:
            print("-" * 20+"Assistant Node"+"-" * 20)
            print(assisant_node[-1])
        elif tool_node:
            print("-" * 20+"Tools Node"+"-" * 20)
            print(tool_node[-1])
```

    --------------------Assistant Node--------------------
    {'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': 'call_I2qudhMCwcw1GzcFN5q80rjj', 'function': {'arguments': '{"a":2,"b":3}', 'name': 'multiply'}, 'type': 'function'}]}, 'response_metadata': {'finish_reason': 'tool_calls', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5'}, 'type': 'ai', 'name': None, 'id': 'run-550e75ad-dbbc-4e55-9f00-aa896228914c', 'example': False, 'tool_calls': [{'name': 'multiply', 'args': {'a': 2, 'b': 3}, 'id': 'call_I2qudhMCwcw1GzcFN5q80rjj', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': None}
    --------------------Tools Node--------------------
    {'content': '6', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'multiply', 'id': '731b7d4f-780d-4a8b-aec9-0d8b9c58c40a', 'tool_call_id': 'call_I2qudhMCwcw1GzcFN5q80rjj', 'artifact': None, 'status': 'success'}
    --------------------Assistant Node--------------------
    {'content': 'The result of multiplying 2 and 3 is 6.', 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5'}, 'type': 'ai', 'name': None, 'id': 'run-2326afa5-eb43-4568-b5ed-424c0a0fa076', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}


#### Forking

Now, let's look at forking.

Let's get the same step as we worked with above, the human input.

Let's create a new thread with our agent.


```python
initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}
thread = await client.threads.create()
async for chunk in client.runs.stream(
    thread["thread_id"],
    assistant_id="agent",
    input=initial_input,
    stream_mode="updates",
):
    if chunk.data:
        assisant_node = chunk.data.get('assistant', {}).get('messages', [])
        tool_node = chunk.data.get('tools', {}).get('messages', [])
        if assisant_node:
            print("-" * 20+"Assistant Node"+"-" * 20)
            print(assisant_node[-1])
        elif tool_node:
            print("-" * 20+"Tools Node"+"-" * 20)
            print(tool_node[-1])
```

    --------------------Assistant Node--------------------
    {'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': 'call_HdWoyLELFZGEcqGxFt2fZzek', 'function': {'arguments': '{"a":2,"b":3}', 'name': 'multiply'}, 'type': 'function'}]}, 'response_metadata': {'finish_reason': 'tool_calls', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5'}, 'type': 'ai', 'name': None, 'id': 'run-cbd081b1-8cef-4ca8-9dd5-aceb134404dc', 'example': False, 'tool_calls': [{'name': 'multiply', 'args': {'a': 2, 'b': 3}, 'id': 'call_HdWoyLELFZGEcqGxFt2fZzek', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': None}
    --------------------Tools Node--------------------
    {'content': '6', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'multiply', 'id': '11dd4a7f-0b6b-44da-b9a4-65f1677c8813', 'tool_call_id': 'call_HdWoyLELFZGEcqGxFt2fZzek', 'artifact': None, 'status': 'success'}
    --------------------Assistant Node--------------------
    {'content': 'The result of multiplying 2 and 3 is 6.', 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5'}, 'type': 'ai', 'name': None, 'id': 'run-936cf990-9302-45c7-9051-6ff1e2e9f316', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}



```python
states = await client.threads.get_history(thread['thread_id'])
to_fork = states[-2]
to_fork['values']
```




    {'messages': [{'content': 'Multiply 2 and 3',
       'additional_kwargs': {'example': False,
        'additional_kwargs': {},
        'response_metadata': {}},
       'response_metadata': {},
       'type': 'human',
       'name': None,
       'id': '93c18b95-9050-4a52-99b8-9374e98ee5db',
       'example': False}]}




```python
to_fork['values']['messages'][0]['id']
```




    '93c18b95-9050-4a52-99b8-9374e98ee5db'




```python
to_fork['next']
```




    ['assistant']




```python
to_fork['checkpoint_id']
```




    '1ef6a44b-27ec-681c-8000-ff7e345aee7e'



Let's edit the state.

Remember how our reducer on `messages` works: 

* It will append, unless we supply a message ID.
* We supply the message ID to overwrite the message, rather than appending to state!


```python
forked_input = {"messages": HumanMessage(content="Multiply 3 and 3",
                                         id=to_fork['values']['messages'][0]['id'])}

forked_config = await client.threads.update_state(
    thread["thread_id"],
    forked_input,
    checkpoint_id=to_fork['checkpoint_id']
)
```


```python
forked_config
```




    {'configurable': {'thread_id': 'c99502e7-b0d7-473e-8295-1ad60e2b7ed2',
      'checkpoint_ns': '',
      'checkpoint_id': '1ef6a44b-90dc-68c8-8001-0c36898e0f34'},
     'checkpoint_id': '1ef6a44b-90dc-68c8-8001-0c36898e0f34'}




```python
states = await client.threads.get_history(thread['thread_id'])
states[0]
```




    {'values': {'messages': [{'content': 'Multiply 3 and 3',
        'additional_kwargs': {'additional_kwargs': {},
         'response_metadata': {},
         'example': False},
        'response_metadata': {},
        'type': 'human',
        'name': None,
        'id': '93c18b95-9050-4a52-99b8-9374e98ee5db',
        'example': False}]},
     'next': ['assistant'],
     'tasks': [{'id': 'da5d6548-62ca-5e69-ba70-f6179b2743bd',
       'name': 'assistant',
       'error': None,
       'interrupts': [],
       'state': None}],
     'metadata': {'step': 1,
      'source': 'update',
      'writes': {'__start__': {'messages': {'id': '93c18b95-9050-4a52-99b8-9374e98ee5db',
         'name': None,
         'type': 'human',
         'content': 'Multiply 3 and 3',
         'example': False,
         'additional_kwargs': {},
         'response_metadata': {}}}},
      'parents': {},
      'graph_id': 'agent'},
     'created_at': '2024-09-03T22:34:46.678333+00:00',
     'checkpoint_id': '1ef6a44b-90dc-68c8-8001-0c36898e0f34',
     'parent_checkpoint_id': '1ef6a44b-27ec-681c-8000-ff7e345aee7e'}



To rerun, we pass in the `checkpoint_id`.


```python
async for chunk in client.runs.stream(
    thread["thread_id"],
    assistant_id="agent",
    input=None,
    stream_mode="updates",
    checkpoint_id=forked_config['checkpoint_id']
):
    if chunk.data:
        assisant_node = chunk.data.get('assistant', {}).get('messages', [])
        tool_node = chunk.data.get('tools', {}).get('messages', [])
        if assisant_node:
            print("-" * 20+"Assistant Node"+"-" * 20)
            print(assisant_node[-1])
        elif tool_node:
            print("-" * 20+"Tools Node"+"-" * 20)
            print(tool_node[-1])
```

    --------------------Assistant Node--------------------
    {'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': 'call_aodhCt5fWv33qVbO7Nsub9Q3', 'function': {'arguments': '{"a":3,"b":3}', 'name': 'multiply'}, 'type': 'function'}]}, 'response_metadata': {'finish_reason': 'tool_calls', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5'}, 'type': 'ai', 'name': None, 'id': 'run-e9759422-e537-4b9b-b583-36c688e13b4b', 'example': False, 'tool_calls': [{'name': 'multiply', 'args': {'a': 3, 'b': 3}, 'id': 'call_aodhCt5fWv33qVbO7Nsub9Q3', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': None}
    --------------------Tools Node--------------------
    {'content': '9', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'multiply', 'id': '89787b0b-93de-4c0a-bea8-d2c3845534e1', 'tool_call_id': 'call_aodhCt5fWv33qVbO7Nsub9Q3', 'artifact': None, 'status': 'success'}
    --------------------Assistant Node--------------------
    {'content': 'The result of multiplying 3 by 3 is 9.', 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5'}, 'type': 'ai', 'name': None, 'id': 'run-0e16610f-4e8d-46f3-a5df-c2f187fae593', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}


### LangGraph Studio

Let's look at forking in the Studio UI with our `agent`, which uses `module-1/studio/agent.py` set in `module-1/studio/langgraph.json`.
