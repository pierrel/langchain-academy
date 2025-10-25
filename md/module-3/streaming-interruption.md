[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-3/streaming-interruption.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239464-lesson-1-streaming)

# Streaming

## Review

In module 2, covered a few ways to customize graph state and memory.
 
We built up to a Chatbot with external memory that can sustain long-running conversations. 

## Goals

This module will dive into `human-in-the-loop`, which builds on memory and allows users to interact directly with graphs in various ways. 

To set the stage for `human-in-the-loop`, we'll first dive into streaming, which provides several ways to visualize graph output (e.g., node state or chat model tokens) over the course of execution.


```python
%%capture --no-stderr
%pip install --quiet -U langgraph langchain_openai langgraph_sdk
```

## Streaming

LangGraph is built with [first class support for streaming](https://langchain-ai.github.io/langgraph/concepts/low_level/#streaming).

Let's set up our Chatbot from Module 2, and show various way to stream outputs from the graph during execution. 


```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
```

Note that we use `RunnableConfig` with `call_model` to enable token-wise streaming. This is [only needed with python < 3.11](https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/). We include in case you are running this notebook in CoLab, which will use python 3.x. 


```python
from IPython.display import Image, display
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState

# LLM
model = ChatOpenAI(model="gpt-4o", temperature=0) 

# State 
class State(MessagesState):
    summary: str

# Define the logic to call the model
def call_model(state: State, config: RunnableConfig):
    
    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, then we add it
    if summary:
        
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]
    
    else:
        messages = state["messages"]
    
    response = model.invoke(messages, config)
    return {"messages": response}

def summarize_conversation(state: State):
    
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt 
    if summary:
        
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

# Determine whether to end or summarize the conversation
def should_continue(state: State)-> Literal ["summarize_conversation",END]:
    
    """Return the next node to execute."""
    
    messages = state["messages"]
    
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return END

# Define a new graph
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# Compile
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
display(Image(graph.get_graph().draw_mermaid_png()))
```


    
![png](streaming-interruption_files/streaming-interruption_6_0.png)
    


### Streaming full state

Now, let's talk about ways to [stream our graph state](https://langchain-ai.github.io/langgraph/concepts/low_level/#streaming).

`.stream` and `.astream` are sync and async methods for streaming back results. 
 
LangGraph supports a few [different streaming modes](https://langchain-ai.github.io/langgraph/how-tos/stream-values/) for [graph state](https://langchain-ai.github.io/langgraph/how-tos/stream-values/):
 
* `values`: This streams the full state of the graph after each node is called.
* `updates`: This streams updates to the state of the graph after each node is called.

![values_vs_updates.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbaf892d24625a201744e5_streaming1.png)

Let's look at `stream_mode="updates"`.

Because we stream with `updates`, we only see updates to the state after node in the graph is run.

Each `chunk` is a dict with `node_name` as the key and the updated state as the value.


```python
# Create a thread
config = {"configurable": {"thread_id": "1"}}

# Start conversation
for chunk in graph.stream({"messages": [HumanMessage(content="hi! I'm Lance")]}, config, stream_mode="updates"):
    print(chunk)
```

    {'conversation': {'messages': AIMessage(content='Hello Lance! How can I assist you today?', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 11, 'total_tokens': 21, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_f64f290af2', 'id': 'chatcmpl-CSqvbTb4Lar7zEfryASuhaZVBnKOw', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--b612e04b-3ff6-4b47-8b88-628f35e44aa7-0', usage_metadata={'input_tokens': 11, 'output_tokens': 10, 'total_tokens': 21, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})}}


Let's now just print the state update.


```python
# Start conversation
for chunk in graph.stream({"messages": [HumanMessage(content="hi! I'm Lance")]}, config, stream_mode="updates"):
    chunk['conversation']["messages"].pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    
    Hi Lance! How's it going? What can I do for you today?


Now, we can see `stream_mode="values"`.

This is the `full state` of the graph after the `conversation` node is called.


```python
# Start conversation, again
config = {"configurable": {"thread_id": "2"}}

# Start conversation
input_message = HumanMessage(content="hi! I'm Lance")
for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    for m in event['messages']:
        m.pretty_print()
    print("---"*25)
```

    ================================[1m Human Message [0m=================================
    
    hi! I'm Lance
    ---------------------------------------------------------------------------
    ================================[1m Human Message [0m=================================
    
    hi! I'm Lance
    ==================================[1m Ai Message [0m==================================
    
    Hello, Lance! How can I assist you today?
    ---------------------------------------------------------------------------


### Streaming tokens

We often want to stream more than graph state.

In particular, with chat model calls it is common to stream the tokens as they are generated.

We can do this [using the `.astream_events` method](https://langchain-ai.github.io/langgraph/how-tos/streaming-from-final-node/#stream-outputs-from-the-final-node), which streams back events as they happen inside nodes!

Each event is a dict with a few keys:
 
* `event`: This is the type of event that is being emitted. 
* `name`: This is the name of event.
* `data`: This is the data associated with the event.
* `metadata`: Contains`langgraph_node`, the node emitting the event.

Let's have a look.


```python
config = {"configurable": {"thread_id": "3"}}
input_message = HumanMessage(content="Tell me about the 49ers NFL team")
async for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):
    print(f"Node: {event['metadata'].get('langgraph_node','')}. Type: {event['event']}. Name: {event['name']}")
```

    Node: . Type: on_chain_start. Name: LangGraph
    Node: conversation. Type: on_chain_start. Name: conversation
    Node: conversation. Type: on_chat_model_start. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    Node: conversation. Type: on_chat_model_end. Name: ChatOpenAI
    Node: conversation. Type: on_chain_start. Name: should_continue
    Node: conversation. Type: on_chain_end. Name: should_continue
    Node: conversation. Type: on_chain_stream. Name: conversation
    Node: conversation. Type: on_chain_end. Name: conversation
    Node: . Type: on_chain_stream. Name: LangGraph
    Node: . Type: on_chain_end. Name: LangGraph


The central point is that tokens from chat models within your graph have the `on_chat_model_stream` type.

We can use `event['metadata']['langgraph_node']` to select the node to stream from.

And we can use `event['data']` to get the actual data for each event, which in this case is an `AIMessageChunk`. 


```python
node_to_stream = 'conversation'
config = {"configurable": {"thread_id": "4"}}
input_message = HumanMessage(content="Tell me about the 49ers NFL team")
async for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):
    # Get chat model tokens from a particular node 
    if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node','') == node_to_stream:
        print(event["data"])
```

    {'chunk': AIMessageChunk(content='', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='The', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' San', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Francisco', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' ', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='49', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='ers', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' are', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' a', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' professional', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' American', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' football', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' team', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' based', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' in', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' the', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' San', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Francisco', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Bay', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Area', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='.', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' They', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' compete', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' in', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' the', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' National', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Football', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' League', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' (', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='NFL', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=')', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' as', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' a', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' member', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' club', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' of', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' the', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' league', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content="'s", additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' National', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Football', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Conference', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' (', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='N', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='FC', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=')', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' West', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' division', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='.', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' The', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' team', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' was', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' founded', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' in', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' ', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='194', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='6', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' as', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' a', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' charter', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' member', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' of', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' the', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' All', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='-Amer', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='ica', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Football', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Conference', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' (', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='AA', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='FC', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=')', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' and', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' joined', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' the', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' NFL', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' in', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' ', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='194', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='9', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' when', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' the', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' leagues', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' merged', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='.\n\n', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='###', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Key', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Points', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=':\n\n', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='-', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' **', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='St', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='adium', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='**', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=':', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' The', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' ', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='49', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='ers', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' play', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' their', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' home', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' games', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' at', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Levi', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content="'s", additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Stadium', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' in', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Santa', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Clara', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' California', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' which', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' they', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' moved', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' to', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' in', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' ', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='201', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='4', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='.', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Before', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' that', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' they', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' played', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' at', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Cand', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='lestick', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Park', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' in', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' San', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Francisco', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='.\n\n', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='-', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' **', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='Team', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Colors', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' and', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Masc', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='ot', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='**', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=':', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' The', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=" team's", additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' colors', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' are', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' red', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' gold', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' and', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' white', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' and', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' their', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' mascot', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' is', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' S', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='ourd', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='ough', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Sam', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='.\n\n', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='-', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' **', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='Champ', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='ionship', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='s', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='**', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=':', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' The', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' ', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='49', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='ers', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' have', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' won', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' five', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Super', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Bowl', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' championships', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' (', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='Super', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Bowl', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' XVI', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' XIX', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' XX', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='III', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' XX', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='IV', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' and', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' XX', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='IX', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='),', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' with', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' their', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' most', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' successful', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' period', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' being', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' the', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' ', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='198', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='0', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='s', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' and', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' early', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' ', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='199', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='0', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='s', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='.\n\n', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='-', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' **', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='Not', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='able', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Figures', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='**', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=':', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' The', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' team', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' has', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' had', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' several', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Hall', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' of', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Fame', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' players', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' including', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Joe', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Montana', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Jerry', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Rice', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Steve', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Young', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Ronnie', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' L', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='ott', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' and', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Charles', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Haley', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='.', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Bill', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Walsh', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' the', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' legendary', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' head', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' coach', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' is', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' credited', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' with', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' developing', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' the', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' West', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Coast', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' offense', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' which', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' became', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' a', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' staple', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' of', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' the', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=" team's", additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' success', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='.\n\n', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='-', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' **', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='R', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='ival', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='ries', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='**', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=':', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' The', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' ', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='49', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='ers', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' have', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' notable', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' rival', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='ries', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' with', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' the', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Seattle', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Seahawks', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Los', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Angeles', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Rams', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' and', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' historically', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' with', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' the', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Dallas', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Cowboys', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' and', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Green', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Bay', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Packers', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='.\n\n', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='-', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' **', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='Recent', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Performance', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='**', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=':', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' In', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' recent', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' years', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' the', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' ', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='49', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='ers', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' have', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' experienced', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' a', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' resurgence', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' including', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' a', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Super', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Bowl', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' appearance', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' in', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' the', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' ', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='201', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='9', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' season', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' (', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='Super', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Bowl', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' LIV', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='),', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' where', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' they', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' were', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' defeated', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' by', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' the', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Kansas', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' City', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' Chiefs', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='.\n\n', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='The', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' ', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='49', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='ers', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' are', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' known', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' for', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' their', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' rich', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' history', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' passionate', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' fan', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' base', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=',', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' and', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' significant', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' contributions', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' to', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' the', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' NFL', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content="'s", additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' development', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' and', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content=' popularity', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='.', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8')}
    {'chunk': AIMessageChunk(content='', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_eb3c3cb84d', 'service_tier': 'default', 'model_provider': 'openai'}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8', chunk_position='last')}
    {'chunk': AIMessageChunk(content='', additional_kwargs={}, response_metadata={}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8', usage_metadata={'input_tokens': 16, 'output_tokens': 382, 'total_tokens': 398, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})}
    {'chunk': AIMessageChunk(content='', additional_kwargs={}, response_metadata={}, id='lc_run--2dc940a7-76b1-4cb1-8e3c-4c18bd4598a8', chunk_position='last')}


As you see above, just use the `chunk` key to get the `AIMessageChunk`.


```python
config = {"configurable": {"thread_id": "5"}}
input_message = HumanMessage(content="Tell me about the 49ers NFL team")
async for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):
    # Get chat model tokens from a particular node 
    if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node','') == node_to_stream:
        data = event["data"]
        print(data["chunk"].content, end="|")
```

    |The| San| Francisco| |49|ers| are| a| professional| American| football| team| based| in| the| San| Francisco| Bay| Area|.| They| compete| in| the| National| Football| League| (|NFL|)| as| a| member| club| of| the| league|'s| National| Football| Conference| (|N|FC|)| West| division|.| The| team| was| founded| in| |194|6| as| a| charter| member| of| the| All|-Amer|ica| Football| Conference| (|AA|FC|)| and| joined| the| NFL| in| |194|9| when| the| leagues| merged|.
    
    |The| |49|ers| have| a| rich| history| and| are| one| of| the| most| successful| teams| in| NFL| history|.| They| have| won| five| Super| Bowl| championships|,| with| victories| in| Super| Bow|ls| XVI|,| XIX|,| XX|III|,| XX|IV|,| and| XX|IX|.| The| team| has| also| won| numerous| division| titles| and| made| several| playoff| appearances|.
    
    |The| |49|ers| are| known| for| their| iconic| players| and| coaches|,| including| Hall| of| Fam|ers| like| Joe| Montana|,| Jerry| Rice|,| Steve| Young|,| Ronnie| L|ott|,| and| coach| Bill| Walsh|,| who| is| credited| with| popular|izing| the| West| Coast| offense|.| The| team's| colors| are| red| and| gold|,| and| they| play| their| home| games| at| Levi|'s| Stadium| in| Santa| Clara|,| California|,| which| they| moved| to| in| |201|4| after| previously| playing| at| Cand|lestick| Park| in| San| Francisco|.
    
    |The| |49|ers| have| a| passionate| fan| base| and| a| stor|ied| rivalry| with| teams| like| the| Dallas| Cowboys|,| Green| Bay| Packers|,| and| Seattle| Seahawks|.| The| team's| success| in| the| |198|0|s| and| |199|0|s|,| particularly| under| the| leadership| of| Bill| Walsh| and| George| Se|if|ert|,| helped| establish| them| as| one| of| the| premier| franchises| in| the| NFL|.||||

### Streaming with LangGraph API

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

The LangGraph API [supports editing graph state](https://langchain-ai.github.io/langgraph/cloud/how-tos/human_in_the_loop_edit_state/#initial-invocation). 


```python
if 'google.colab' in str(get_ipython()):
    raise Exception("Unfortunately LangGraph Studio is currently not supported on Google Colab")
```


```python
from langgraph_sdk import get_client

# This is the URL of the local development server
URL = "http://127.0.0.1:2024"
client = get_client(url=URL)

# Search all hosted graphs
assistants = await client.assistants.search()
```

Let's [stream `values`](https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_values/), like before.


```python
# Create a new thread
thread = await client.threads.create()
# Input message
input_message = HumanMessage(content="Multiply 2 and 3")
async for event in client.runs.stream(thread["thread_id"], 
                                      assistant_id="agent", 
                                      input={"messages": [input_message]}, 
                                      stream_mode="values"):
    print(event)
```

    StreamPart(event='metadata', data={'run_id': '019a0358-31b4-7143-af47-2feeac0b27ce', 'attempt': 1})
    StreamPart(event='values', data={'messages': [{'content': 'Multiply 2 and 3', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '9aaa247f-1e6e-4451-af25-ac678fe46d82'}]})
    StreamPart(event='values', data={'messages': [{'content': 'Multiply 2 and 3', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '9aaa247f-1e6e-4451-af25-ac678fe46d82'}, {'content': '', 'additional_kwargs': {'refusal': None}, 'response_metadata': {'token_usage': {'completion_tokens': 17, 'prompt_tokens': 134, 'total_tokens': 151, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_eb3c3cb84d', 'id': 'chatcmpl-CSqw4HhOXHahA4tIr4mIdEhQB1QB4', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, 'type': 'ai', 'name': None, 'id': 'lc_run--4a3794f6-52c3-41b5-9620-1710d6e8392d-0', 'tool_calls': [{'name': 'multiply', 'args': {'a': 2, 'b': 3}, 'id': 'call_3sCWZZ89AoUe91MYc3ZBtJ0P', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 134, 'output_tokens': 17, 'total_tokens': 151, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}}]})
    StreamPart(event='values', data={'messages': [{'content': 'Multiply 2 and 3', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '9aaa247f-1e6e-4451-af25-ac678fe46d82'}, {'content': '', 'additional_kwargs': {'refusal': None}, 'response_metadata': {'token_usage': {'completion_tokens': 17, 'prompt_tokens': 134, 'total_tokens': 151, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_eb3c3cb84d', 'id': 'chatcmpl-CSqw4HhOXHahA4tIr4mIdEhQB1QB4', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, 'type': 'ai', 'name': None, 'id': 'lc_run--4a3794f6-52c3-41b5-9620-1710d6e8392d-0', 'tool_calls': [{'name': 'multiply', 'args': {'a': 2, 'b': 3}, 'id': 'call_3sCWZZ89AoUe91MYc3ZBtJ0P', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 134, 'output_tokens': 17, 'total_tokens': 151, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}}, {'content': '6', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'multiply', 'id': 'f05da6b3-27be-4896-967a-3ff60aa06d85', 'tool_call_id': 'call_3sCWZZ89AoUe91MYc3ZBtJ0P', 'artifact': None, 'status': 'success'}]})
    StreamPart(event='values', data={'messages': [{'content': 'Multiply 2 and 3', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '9aaa247f-1e6e-4451-af25-ac678fe46d82'}, {'content': '', 'additional_kwargs': {'refusal': None}, 'response_metadata': {'token_usage': {'completion_tokens': 17, 'prompt_tokens': 134, 'total_tokens': 151, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_eb3c3cb84d', 'id': 'chatcmpl-CSqw4HhOXHahA4tIr4mIdEhQB1QB4', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, 'type': 'ai', 'name': None, 'id': 'lc_run--4a3794f6-52c3-41b5-9620-1710d6e8392d-0', 'tool_calls': [{'name': 'multiply', 'args': {'a': 2, 'b': 3}, 'id': 'call_3sCWZZ89AoUe91MYc3ZBtJ0P', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 134, 'output_tokens': 17, 'total_tokens': 151, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}}, {'content': '6', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'multiply', 'id': 'f05da6b3-27be-4896-967a-3ff60aa06d85', 'tool_call_id': 'call_3sCWZZ89AoUe91MYc3ZBtJ0P', 'artifact': None, 'status': 'success'}, {'content': 'The result of multiplying 2 and 3 is 6.', 'additional_kwargs': {'refusal': None}, 'response_metadata': {'token_usage': {'completion_tokens': 14, 'prompt_tokens': 159, 'total_tokens': 173, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_eb3c3cb84d', 'id': 'chatcmpl-CSqw4xk8t3sPODL5beks5TlJBozgB', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, 'type': 'ai', 'name': None, 'id': 'lc_run--33a25ab0-f748-4c7f-a086-d9249e25fdc0-0', 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 159, 'output_tokens': 14, 'total_tokens': 173, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}}]})


The streamed objects have: 

* `event`: Type
* `data`: State


```python
from langchain_core.messages import convert_to_messages
thread = await client.threads.create()
input_message = HumanMessage(content="Multiply 2 and 3")
async for event in client.runs.stream(thread["thread_id"], assistant_id="agent", input={"messages": [input_message]}, stream_mode="values"):
    messages = event.data.get('messages',None)
    if messages:
        print(convert_to_messages(messages)[-1])
    print('='*25)
```

    =========================
    content='Multiply 2 and 3' additional_kwargs={} response_metadata={} id='c3ec872a-99a1-4eec-bcb6-a04973f48ac5'
    =========================
    content='' additional_kwargs={'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 134, 'output_tokens': 17, 'total_tokens': 151, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}, 'refusal': None} response_metadata={'token_usage': {'completion_tokens': 17, 'prompt_tokens': 134, 'total_tokens': 151, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_f64f290af2', 'id': 'chatcmpl-CSqw6HYoyCI7z2AuKAAfSTQGbvzla', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None} id='lc_run--c91028e7-7a0a-4746-a4f5-edcff5380abc-0' tool_calls=[{'name': 'multiply', 'args': {'a': 2, 'b': 3}, 'id': 'call_AFChrxIQGbr7mmzr8buxymY0', 'type': 'tool_call'}]
    =========================
    content='6' name='multiply' id='f69a844b-5f82-4256-96dd-92b044e888d9' tool_call_id='call_AFChrxIQGbr7mmzr8buxymY0'
    =========================
    content='The result of multiplying 2 and 3 is 6.' additional_kwargs={'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 159, 'output_tokens': 14, 'total_tokens': 173, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}, 'refusal': None} response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 159, 'total_tokens': 173, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_eb3c3cb84d', 'id': 'chatcmpl-CSqw7xBeinGuHlx0upkmo2tryppco', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--dba1c4af-ed8f-4fed-8770-2c3429603cd0-0'
    =========================


There are some new streaming mode that are only supported via the API.

For example, we can [use `messages` mode](https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages/) to better handle the above case!

This mode currently assumes that you have a `messages` key in your graph, which is a list of messages.

All events emitted using `messages` mode have two attributes:

* `event`: This is the name of the event
* `data`: This is data associated with the event


```python
thread = await client.threads.create()
input_message = HumanMessage(content="Multiply 2 and 3")
async for event in client.runs.stream(thread["thread_id"], 
                                      assistant_id="agent", 
                                      input={"messages": [input_message]}, 
                                      stream_mode="messages"):
    print(event.event)
```

    metadata
    messages/metadata
    messages/partial
    messages/partial
    messages/partial
    messages/partial
    messages/partial
    messages/partial
    messages/partial
    messages/partial
    messages/partial
    messages/partial
    messages/partial
    messages/partial
    messages/partial
    messages/metadata
    messages/complete
    messages/metadata
    messages/partial
    messages/partial
    messages/partial
    messages/partial
    messages/partial
    messages/partial
    messages/partial
    messages/partial
    messages/partial
    messages/partial
    messages/partial
    messages/partial
    messages/partial
    messages/partial
    messages/partial
    messages/partial
    messages/partial


We can see a few events: 

* `metadata`: metadata about the run
* `messages/complete`: fully formed message 
* `messages/partial`: chat model tokens

You can dig further into the types [here](https://langchain-ai.github.io/langgraph/cloud/concepts/api/#modemessages).

Now, let's show how to stream these messages. 

We'll define a helper function for better formatting of the tool calls in messages.


```python
thread = await client.threads.create()
input_message = HumanMessage(content="Multiply 2 and 3")

def format_tool_calls(tool_calls):
    """
    Format a list of tool calls into a readable string.

    Args:
        tool_calls (list): A list of dictionaries, each representing a tool call.
            Each dictionary should have 'id', 'name', and 'args' keys.

    Returns:
        str: A formatted string of tool calls, or "No tool calls" if the list is empty.

    """

    if tool_calls:
        formatted_calls = []
        for call in tool_calls:
            formatted_calls.append(
                f"Tool Call ID: {call['id']}, Function: {call['name']}, Arguments: {call['args']}"
            )
        return "\n".join(formatted_calls)
    return "No tool calls"

async for event in client.runs.stream(
    thread["thread_id"],
    assistant_id="agent",
    input={"messages": [input_message]},
    stream_mode="messages",):
    
    # Handle metadata events
    if event.event == "metadata":
        print(f"Metadata: Run ID - {event.data['run_id']}")
        print("-" * 50)
    
    # Handle partial message events
    elif event.event == "messages/partial":
        for data_item in event.data:
            # Process user messages
            if "role" in data_item and data_item["role"] == "user":
                print(f"Human: {data_item['content']}")
            else:
                # Extract relevant data from the event
                tool_calls = data_item.get("tool_calls", [])
                invalid_tool_calls = data_item.get("invalid_tool_calls", [])
                content = data_item.get("content", "")
                response_metadata = data_item.get("response_metadata", {})

                if content:
                    print(f"AI: {content}")

                if tool_calls:
                    print("Tool Calls:")
                    print(format_tool_calls(tool_calls))

                if invalid_tool_calls:
                    print("Invalid Tool Calls:")
                    print(format_tool_calls(invalid_tool_calls))

                if response_metadata and response_metadata.get("finish_reason"):
                    print(f"Response Metadata: Finish Reason - {response_metadata['finish_reason']}")                    
        print("-" * 50)
```

    Metadata: Run ID - 019a0358-57dc-76f9-bc63-633eee467a86
    --------------------------------------------------
    Tool Calls:
    Tool Call ID: call_mDtKBiuzkpN5ITWykhEFN0XU, Function: multiply, Arguments: {}
    --------------------------------------------------
    Tool Calls:
    Tool Call ID: call_mDtKBiuzkpN5ITWykhEFN0XU, Function: multiply, Arguments: {}
    --------------------------------------------------
    Tool Calls:
    Tool Call ID: call_mDtKBiuzkpN5ITWykhEFN0XU, Function: multiply, Arguments: {}
    --------------------------------------------------
    Tool Calls:
    Tool Call ID: call_mDtKBiuzkpN5ITWykhEFN0XU, Function: multiply, Arguments: {}
    --------------------------------------------------
    Tool Calls:
    Tool Call ID: call_mDtKBiuzkpN5ITWykhEFN0XU, Function: multiply, Arguments: {'a': 2}
    --------------------------------------------------
    Tool Calls:
    Tool Call ID: call_mDtKBiuzkpN5ITWykhEFN0XU, Function: multiply, Arguments: {'a': 2}
    --------------------------------------------------
    Tool Calls:
    Tool Call ID: call_mDtKBiuzkpN5ITWykhEFN0XU, Function: multiply, Arguments: {'a': 2}
    --------------------------------------------------
    Tool Calls:
    Tool Call ID: call_mDtKBiuzkpN5ITWykhEFN0XU, Function: multiply, Arguments: {'a': 2}
    --------------------------------------------------
    Tool Calls:
    Tool Call ID: call_mDtKBiuzkpN5ITWykhEFN0XU, Function: multiply, Arguments: {'a': 2, 'b': 3}
    --------------------------------------------------
    Tool Calls:
    Tool Call ID: call_mDtKBiuzkpN5ITWykhEFN0XU, Function: multiply, Arguments: {'a': 2, 'b': 3}
    --------------------------------------------------
    Tool Calls:
    Tool Call ID: call_mDtKBiuzkpN5ITWykhEFN0XU, Function: multiply, Arguments: {'a': 2, 'b': 3}
    Response Metadata: Finish Reason - tool_calls
    --------------------------------------------------
    Tool Calls:
    Tool Call ID: call_mDtKBiuzkpN5ITWykhEFN0XU, Function: multiply, Arguments: {'a': 2, 'b': 3}
    Response Metadata: Finish Reason - tool_calls
    --------------------------------------------------
    Tool Calls:
    Tool Call ID: call_mDtKBiuzkpN5ITWykhEFN0XU, Function: multiply, Arguments: {'a': 2, 'b': 3}
    Response Metadata: Finish Reason - tool_calls
    --------------------------------------------------
    --------------------------------------------------
    AI: The
    --------------------------------------------------
    AI: The result
    --------------------------------------------------
    AI: The result of
    --------------------------------------------------
    AI: The result of multiplying
    --------------------------------------------------
    AI: The result of multiplying 
    --------------------------------------------------
    AI: The result of multiplying 2
    --------------------------------------------------
    AI: The result of multiplying 2 and
    --------------------------------------------------
    AI: The result of multiplying 2 and 
    --------------------------------------------------
    AI: The result of multiplying 2 and 3
    --------------------------------------------------
    AI: The result of multiplying 2 and 3 is
    --------------------------------------------------
    AI: The result of multiplying 2 and 3 is 
    --------------------------------------------------
    AI: The result of multiplying 2 and 3 is 6
    --------------------------------------------------
    AI: The result of multiplying 2 and 3 is 6.
    --------------------------------------------------
    AI: The result of multiplying 2 and 3 is 6.
    Response Metadata: Finish Reason - stop
    --------------------------------------------------
    AI: The result of multiplying 2 and 3 is 6.
    Response Metadata: Finish Reason - stop
    --------------------------------------------------
    AI: The result of multiplying 2 and 3 is 6.
    Response Metadata: Finish Reason - stop
    --------------------------------------------------



```python

```
