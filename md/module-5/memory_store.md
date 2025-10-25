# Chatbot with Memory

## Review

[Memory](https://pmc.ncbi.nlm.nih.gov/articles/PMC10410470/) is a cognitive function that allows people to store, retrieve, and use information to understand their present and future. 

There are [various long-term memory types](https://langchain-ai.github.io/langgraph/concepts/memory/#memory) that can be used in AI applications.

## Goals

Here, we'll introduce the [LangGraph Memory Store](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) as a way to save and retrieve long-term memories.

We'll build a chatbot that uses both `short-term (within-thread)` and `long-term (across-thread)` memory.
 
We'll focus on long-term [semantic memory](https://langchain-ai.github.io/langgraph/concepts/memory/#semantic-memory), which will be facts about the user. 

These long-term memories will be used to create a personalized chatbot that can remember facts about the user.

It will save memory ["in the hot path"](https://langchain-ai.github.io/langgraph/concepts/memory/#writing-memories), as the user is chatting with it.


```python
%%capture --no-stderr
%pip install -U langchain_openai langgraph langchain_core
```

We'll use [LangSmith](https://docs.smith.langchain.com/) for [tracing](https://docs.smith.langchain.com/concepts/tracing).


```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
```

## Introduction to the LangGraph Store

The [LangGraph Memory Store](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) provides a way to store and retrieve information *across threads* in LangGraph.

This is an  [open source base class](https://blog.langchain.dev/launching-long-term-memory-support-in-langgraph/) for persistent `key-value` stores.


```python
import uuid
from langgraph.store.memory import InMemoryStore
in_memory_store = InMemoryStore()
```

When storing objects (e.g., memories) in the [Store](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore), we provide:

- The `namespace` for the object, a tuple (similar to directories)
- the object `key` (similar to filenames)
- the object `value` (similar to file contents)

We use the [put](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore.put) method to save an object to the store by `namespace` and `key`.

![langgraph_store.png](memory_store_files/6281b4e3-4930-467e-83ce-ba1aa837ca16.png)


```python
# Namespace for the memory to save
user_id = "1"
namespace_for_memory = (user_id, "memories")

# Save a memory to namespace as key and value
key = str(uuid.uuid4())

# The value needs to be a dictionary  
value = {"food_preference" : "I like pizza"}

# Save the memory
in_memory_store.put(namespace_for_memory, key, value)
```

We use [search](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore.search) to retrieve objects from the store by `namespace`.

This returns a list.


```python
# Search 
memories = in_memory_store.search(namespace_for_memory)
type(memories)
```




    list




```python
# Metatdata 
memories[0].dict()
```




    {'value': {'food_preference': 'I like pizza'},
     'key': 'a754b8c5-e8b7-40ec-834b-c426a9a7c7cc',
     'namespace': ['1', 'memories'],
     'created_at': '2024-11-04T22:48:16.727572+00:00',
     'updated_at': '2024-11-04T22:48:16.727574+00:00'}




```python
# The key, value
print(memories[0].key, memories[0].value)
```

    a754b8c5-e8b7-40ec-834b-c426a9a7c7cc {'food_preference': 'I like pizza'}


We can also use [get](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore.get) to retrieve an object by `namespace` and `key`.


```python
# Get the memory by namespace and key
memory = in_memory_store.get(namespace_for_memory, key)
memory.dict()
```




    {'value': {'food_preference': 'I like pizza'},
     'key': 'a754b8c5-e8b7-40ec-834b-c426a9a7c7cc',
     'namespace': ['1', 'memories'],
     'created_at': '2024-11-04T22:48:16.727572+00:00',
     'updated_at': '2024-11-04T22:48:16.727574+00:00'}



## Chatbot with long-term memory

We want a chatbot that [has two types of memory](https://docs.google.com/presentation/d/181mvjlgsnxudQI6S3ritg9sooNyu4AcLLFH1UK0kIuk/edit#slide=id.g30eb3c8cf10_0_156):

1. `Short-term (within-thread) memory`: Chatbot can persist conversational history and / or allow interruptions in a chat session.
2. `Long-term (cross-thread) memory`: Chatbot can remember information about a specific user *across all chat sessions*.


```python
_set_env("OPENAI_API_KEY")
```

For `short-term memory`, we'll use a [checkpointer](https://langchain-ai.github.io/langgraph/concepts/persistence/#checkpointer-libraries). 

See Module 2 and our [conceptual docs](https://langchain-ai.github.io/langgraph/concepts/persistence/) for more on checkpointers, but in summary:

* They write the graph state at each step to a thread.
* They persist the chat history in the thread.
* They allow the graph to be interrupted and / or resumed from any step in the thread.
 
And, for `long-term memory`, we'll use the [LangGraph Store](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) as introduced above.


```python
# Chat model 
from langchain_openai import ChatOpenAI

# Initialize the LLM
model = ChatOpenAI(model="gpt-4o", temperature=0) 
```

The chat history will be saved to short-term memory using the checkpointer.

The chatbot will reflect on the chat history. 

It will then create and save a memory to the [LangGraph Store](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore).

This memory is accessible in future chat sessions to personalize the chatbot's responses.


```python
from IPython.display import Image, display

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

# Chatbot instruction
MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user. 
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}"""

# Create new memory from the chat history and any existing memory
CREATE_MEMORY_INSTRUCTION = """"You are collecting information about the user to personalize your responses.

CURRENT USER INFORMATION:
{memory}

INSTRUCTIONS:
1. Review the chat history below carefully
2. Identify new information about the user, such as:
   - Personal details (name, location)
   - Preferences (likes, dislikes)
   - Interests and hobbies
   - Past experiences
   - Goals or future plans
3. Merge any new information with existing memory
4. Format the memory as a clear, bulleted list
5. If new information conflicts with existing memory, keep the most recent version

Remember: Only include factual information directly stated by the user. Do not make assumptions or inferences.

Based on the chat history below, please update the user information:"""

def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Load memory from the store and use it to personalize the chatbot's response."""
    
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Retrieve memory from the store
    namespace = ("memory", user_id)
    key = "user_memory"
    existing_memory = store.get(namespace, key)

    # Extract the actual memory content if it exists and add a prefix
    if existing_memory:
        # Value is a dictionary with a memory key
        existing_memory_content = existing_memory.value.get('memory')
    else:
        existing_memory_content = "No existing memory found."

    # Format the memory in the system prompt
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=existing_memory_content)
    
    # Respond using memory as well as the chat history
    response = model.invoke([SystemMessage(content=system_msg)]+state["messages"])

    return {"messages": response}

def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and save a memory to the store."""
    
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Retrieve existing memory from the store
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")
        
    # Extract the memory
    if existing_memory:
        existing_memory_content = existing_memory.value.get('memory')
    else:
        existing_memory_content = "No existing memory found."

    # Format the memory in the system prompt
    system_msg = CREATE_MEMORY_INSTRUCTION.format(memory=existing_memory_content)
    new_memory = model.invoke([SystemMessage(content=system_msg)]+state['messages'])

    # Overwrite the existing memory in the store 
    key = "user_memory"

    # Write value as a dictionary with a memory key
    store.put(namespace, key, {"memory": new_memory.content})

# Define the graph
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("write_memory", write_memory)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", "write_memory")
builder.add_edge("write_memory", END)

# Store for long-term (across-thread) memory
across_thread_memory = InMemoryStore()

# Checkpointer for short-term (within-thread) memory
within_thread_memory = MemorySaver()

# Compile the graph with the checkpointer fir and store
graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)

# View
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
```


    
![jpeg](memory_store_files/memory_store_19_0.jpg)
    


When we interact with the chatbot, we supply two things:

1. `Short-term (within-thread) memory`: A `thread ID` for persisting the chat history.
2. `Long-term (cross-thread) memory`: A `user ID` to namespace long-term memories to the user.

Let's see how these work together in practice. 


```python
# We supply a thread ID for short-term (within-thread) memory
# We supply a user ID for long-term (across-thread) memory 
config = {"configurable": {"thread_id": "1", "user_id": "1"}}

# User input 
input_messages = [HumanMessage(content="Hi, my name is Lance")]

# Run the graph
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Hi, my name is Lance
    ==================================[1m Ai Message [0m==================================
    
    Hello, Lance! It's nice to meet you. How can I assist you today?



```python
# User input 
input_messages = [HumanMessage(content="I like to bike around San Francisco")]

# Run the graph
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    I like to bike around San Francisco
    ==================================[1m Ai Message [0m==================================
    
    That sounds like a great way to explore the city, Lance! San Francisco has some beautiful routes and views. Do you have a favorite trail or area you like to bike in?


We're using the `MemorySaver` checkpointer for within-thread memory.

This saves the chat history to the thread.

We can look at the chat history saved to the thread.


```python
thread = {"configurable": {"thread_id": "1"}}
state = graph.get_state(thread).values
for m in state["messages"]: 
    m.pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Hi, my name is Lance
    ==================================[1m Ai Message [0m==================================
    
    Hello, Lance! It's nice to meet you. How can I assist you today?
    ================================[1m Human Message [0m=================================
    
    I like to bike around San Francisco
    ==================================[1m Ai Message [0m==================================
    
    That sounds like a great way to explore the city, Lance! San Francisco has some beautiful routes and views. Do you have a favorite trail or area you like to bike in?


Recall that we compiled the graph with our the store: 

```python
across_thread_memory = InMemoryStore()
```

And, we added a node to the graph (`write_memory`) that reflects on the chat history and saves a memory to the store.

We can to see if the memory was saved to the store.


```python
# Namespace for the memory to save
user_id = "1"
namespace = ("memory", user_id)
existing_memory = across_thread_memory.get(namespace, "user_memory")
existing_memory.dict()
```




    {'value': {'memory': "**Updated User Information:**\n- User's name is Lance.\n- Likes to bike around San Francisco."},
     'key': 'user_memory',
     'namespace': ['memory', '1'],
     'created_at': '2024-11-05T00:12:17.383918+00:00',
     'updated_at': '2024-11-05T00:12:25.469528+00:00'}



Now, let's kick off a *new thread* with the *same user ID*.

We should see that the chatbot remembered the user's profile and used it to personalize the response.


```python
# We supply a user ID for across-thread memory as well as a new thread ID
config = {"configurable": {"thread_id": "2", "user_id": "1"}}

# User input 
input_messages = [HumanMessage(content="Hi! Where would you recommend that I go biking?")]

# Run the graph
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Hi! Where would you recommend that I go biking?
    ==================================[1m Ai Message [0m==================================
    
    Hi Lance! Since you enjoy biking around San Francisco, there are some fantastic routes you might love. Here are a few recommendations:
    
    1. **Golden Gate Park**: This is a classic choice with plenty of trails and beautiful scenery. You can explore the park's many attractions, like the Conservatory of Flowers and the Japanese Tea Garden.
    
    2. **The Embarcadero**: A ride along the Embarcadero offers stunning views of the Bay Bridge and the waterfront. It's a great way to experience the city's vibrant atmosphere.
    
    3. **Marin Headlands**: If you're up for a bit of a challenge, biking across the Golden Gate Bridge to the Marin Headlands offers breathtaking views of the city and the Pacific Ocean.
    
    4. **Presidio**: This area has a network of trails with varying difficulty levels, and you can enjoy views of the Golden Gate Bridge and the bay.
    
    5. **Twin Peaks**: For a more challenging ride, head up to Twin Peaks. The climb is worth it for the panoramic views of the city.
    
    Let me know if you want more details on any of these routes!



```python
# User input 
input_messages = [HumanMessage(content="Great, are there any bakeries nearby that I can check out? I like a croissant after biking.")]

# Run the graph
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Great, are there any bakeries nearby that I can check out? I like a croissant after biking.
    ==================================[1m Ai Message [0m==================================
    
    Absolutely, Lance! Here are a few bakeries in San Francisco where you can enjoy a delicious croissant after your ride:
    
    1. **Tartine Bakery**: Located in the Mission District, Tartine is famous for its pastries, and their croissants are a must-try.
    
    2. **Arsicault Bakery**: This bakery in the Richmond District has been praised for its buttery, flaky croissants. It's a bit of a detour, but worth it!
    
    3. **b. Patisserie**: Situated in Lower Pacific Heights, b. Patisserie offers a variety of pastries, and their croissants are particularly popular.
    
    4. **Le Marais Bakery**: With locations in the Marina and Castro, Le Marais offers a charming French bakery experience with excellent croissants.
    
    5. **Neighbor Bakehouse**: Located in the Dogpatch, this bakery is known for its creative pastries, including some fantastic croissants.
    
    These spots should provide a delightful treat after your biking adventures. Enjoy your ride and your croissant!


## Viewing traces in LangSmith

We can see that the memories are retrieved from the store and supplied as part of the system prompt, as expected:

https://smith.langchain.com/public/10268d64-82ff-434e-ac02-4afa5cc15432/r

## Studio

We can also interact with our chatbot in Studio. 

![Screenshot 2024-10-28 at 10.08.27 AM.png](memory_store_files/afa216f7-4b67-4783-82af-c319e0f512ac.png)


