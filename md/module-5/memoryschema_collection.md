# Chatbot with Collection Schema 

## Review

We extended our chatbot to save semantic memories to a single [user profile](https://langchain-ai.github.io/langgraph/concepts/memory/#profile). 

We also introduced a library, [Trustcall](https://github.com/hinthornw/trustcall), to update this schema with new information. 

## Goals

Sometimes we want to save memories to a [collection](https://docs.google.com/presentation/d/181mvjlgsnxudQI6S3ritg9sooNyu4AcLLFH1UK0kIuk/edit#slide=id.g30eb3c8cf10_0_200) rather than single profile. 

Here we'll update our chatbot to [save memories to a collection](https://langchain-ai.github.io/langgraph/concepts/memory/#collection).

We'll also show how to use [Trustcall](https://github.com/hinthornw/trustcall) to update this collection. 



```python
%%capture --no-stderr
%pip install -U langchain_openai langgraph trustcall langchain_core
```


```python
import os, getpass

def _set_env(var: str):
    # Check if the variable is set in the OS environment
    env_value = os.environ.get(var)
    if not env_value:
        # If not set, prompt the user for input
        env_value = getpass.getpass(f"{var}: ")
    
    # Set the environment variable for the current process
    os.environ[var] = env_value

_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
```

## Defining a collection schema

Instead of storing user information in a fixed profile structure, we'll create a flexible collection schema to store memories about user interactions.

Each memory will be stored as a separate entry with a single `content` field for the main information we want to remember

This approach allows us to build an open-ended collection of memories that can grow and change as we learn more about the user.

We can define a collection schema as a [Pydantic](https://docs.pydantic.dev/latest/) object. 


```python
from pydantic import BaseModel, Field

class Memory(BaseModel):
    content: str = Field(description="The main content of the memory. For example: User expressed interest in learning about French.")

class MemoryCollection(BaseModel):
    memories: list[Memory] = Field(description="A list of memories about the user.")
```


```python
_set_env("OPENAI_API_KEY")
```

We can used LangChain's chat model [chat model](https://python.langchain.com/docs/concepts/chat_models/) interface's [`with_structured_output`](https://python.langchain.com/docs/concepts/structured_outputs/#recommended-usage) method to enforce structured output.


```python
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Initialize the model
model = ChatOpenAI(model="gpt-4o", temperature=0)

# Bind schema to model
model_with_structure = model.with_structured_output(MemoryCollection)

# Invoke the model to produce structured output that matches the schema
memory_collection = model_with_structure.invoke([HumanMessage("My name is Lance. I like to bike.")])
memory_collection.memories
```




    [Memory(content="User's name is Lance."),
     Memory(content='Lance likes to bike.')]



We can use `model_dump()` to serialize a Pydantic model instance into a Python dictionary.


```python
memory_collection.memories[0].model_dump()
```




    {'content': "User's name is Lance."}



Save dictionary representation of each memory to the store. 


```python
import uuid
from langgraph.store.memory import InMemoryStore

# Initialize the in-memory store
in_memory_store = InMemoryStore()

# Namespace for the memory to save
user_id = "1"
namespace_for_memory = (user_id, "memories")

# Save a memory to namespace as key and value
key = str(uuid.uuid4())
value = memory_collection.memories[0].model_dump()
in_memory_store.put(namespace_for_memory, key, value)

key = str(uuid.uuid4())
value = memory_collection.memories[1].model_dump()
in_memory_store.put(namespace_for_memory, key, value)
```

Search for memories in the store. 


```python
# Search 
for m in in_memory_store.search(namespace_for_memory):
    print(m.dict())
```

    {'value': {'content': "User's name is Lance."}, 'key': 'e1c4e5ab-ab0f-4cbb-822d-f29240a983af', 'namespace': ['1', 'memories'], 'created_at': '2024-10-30T21:43:26.893775+00:00', 'updated_at': '2024-10-30T21:43:26.893779+00:00'}
    {'value': {'content': 'Lance likes to bike.'}, 'key': 'e132a1ea-6202-43ac-a9a6-3ecf2c1780a8', 'namespace': ['1', 'memories'], 'created_at': '2024-10-30T21:43:26.893833+00:00', 'updated_at': '2024-10-30T21:43:26.893834+00:00'}


## Updating collection schema

We discussed the challenges with updating a profile schema in the last lesson. 

The same applies for collections! 

We want the ability to update the collection with new memories as well as update existing memories in the collection. 

Now we'll show that [Trustcall](https://github.com/hinthornw/trustcall) can be also used to update a collection. 

This enables both addition of new memories as well as [updating existing memories in the collection](https://github.com/hinthornw/trustcall?tab=readme-ov-file#simultanous-updates--insertions
).

Let's define a new extractor with Trustcall. 

As before, we provide the schema for each memory, `Memory`.  

But, we can supply `enable_inserts=True` to allow the extractor to insert new memories to the collection. 


```python
from trustcall import create_extractor

# Create the extractor
trustcall_extractor = create_extractor(
    model,
    tools=[Memory],
    tool_choice="Memory",
    enable_inserts=True,
)
```


```python
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Instruction
instruction = """Extract memories from the following conversation:"""

# Conversation
conversation = [HumanMessage(content="Hi, I'm Lance."), 
                AIMessage(content="Nice to meet you, Lance."), 
                HumanMessage(content="This morning I had a nice bike ride in San Francisco.")]

# Invoke the extractor
result = trustcall_extractor.invoke({"messages": [SystemMessage(content=instruction)] + conversation})
```


```python
# Messages contain the tool calls
for m in result["messages"]:
    m.pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      Memory (call_Pj4kctFlpg9TgcMBfMH33N30)
     Call ID: call_Pj4kctFlpg9TgcMBfMH33N30
      Args:
        content: Lance had a nice bike ride in San Francisco this morning.



```python
# Responses contain the memories that adhere to the schema
for m in result["responses"]: 
    print(m)
```

    content='Lance had a nice bike ride in San Francisco this morning.'



```python
# Metadata contains the tool call  
for m in result["response_metadata"]: 
    print(m)
```

    {'id': 'call_Pj4kctFlpg9TgcMBfMH33N30'}



```python
# Update the conversation
updated_conversation = [AIMessage(content="That's great, did you do after?"), 
                        HumanMessage(content="I went to Tartine and ate a croissant."),                        
                        AIMessage(content="What else is on your mind?"),
                        HumanMessage(content="I was thinking about my Japan, and going back this winter!"),]

# Update the instruction
system_msg = """Update existing memories and create new ones based on the following conversation:"""

# We'll save existing memories, giving them an ID, key (tool name), and value
tool_name = "Memory"
existing_memories = [(str(i), tool_name, memory.model_dump()) for i, memory in enumerate(result["responses"])] if result["responses"] else None
existing_memories
```




    [('0',
      'Memory',
      {'content': 'Lance had a nice bike ride in San Francisco this morning.'})]




```python
# Invoke the extractor with our updated conversation and existing memories
result = trustcall_extractor.invoke({"messages": updated_conversation, 
                                     "existing": existing_memories})
```


```python
# Messages from the model indicate two tool calls were made
for m in result["messages"]:
    m.pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      Memory (call_vxks0YH1hwUxkghv4f5zdkTr)
     Call ID: call_vxks0YH1hwUxkghv4f5zdkTr
      Args:
        content: Lance had a nice bike ride in San Francisco this morning. He went to Tartine and ate a croissant. He was thinking about his trip to Japan and going back this winter!
      Memory (call_Y4S3poQgFmDfPy2ExPaMRk8g)
     Call ID: call_Y4S3poQgFmDfPy2ExPaMRk8g
      Args:
        content: Lance went to Tartine and ate a croissant. He was thinking about his trip to Japan and going back this winter!



```python
# Responses contain the memories that adhere to the schema
for m in result["responses"]: 
    print(m)
```

    content='Lance had a nice bike ride in San Francisco this morning. He went to Tartine and ate a croissant. He was thinking about his trip to Japan and going back this winter!'
    content='Lance went to Tartine and ate a croissant. He was thinking about his trip to Japan and going back this winter!'


This tells us that we updated the first memory in the collection by specifying the `json_doc_id`. 


```python
# Metadata contains the tool call  
for m in result["response_metadata"]: 
    print(m)
```

    {'id': 'call_vxks0YH1hwUxkghv4f5zdkTr', 'json_doc_id': '0'}
    {'id': 'call_Y4S3poQgFmDfPy2ExPaMRk8g'}


LangSmith trace: 

https://smith.langchain.com/public/ebc1cb01-f021-4794-80c0-c75d6ea90446/r

## Chatbot with collection schema updating

Now, let's bring Trustcall into our chatbot to create and update a memory collection.


```python
from IPython.display import Image, display

import uuid

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import merge_message_runs
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore

# Initialize the model
model = ChatOpenAI(model="gpt-4o", temperature=0)

# Memory schema
class Memory(BaseModel):
    content: str = Field(description="The main content of the memory. For example: User expressed interest in learning about French.")

# Create the Trustcall extractor
trustcall_extractor = create_extractor(
    model,
    tools=[Memory],
    tool_choice="Memory",
    # This allows the extractor to insert new memories
    enable_inserts=True,
)

# Chatbot instruction
MODEL_SYSTEM_MESSAGE = """You are a helpful chatbot. You are designed to be a companion to a user. 

You have a long term memory which keeps track of information you learn about the user over time.

Current Memory (may include updated memories from this conversation): 

{memory}"""

# Trustcall instruction
TRUSTCALL_INSTRUCTION = """Reflect on following interaction. 

Use the provided tools to retain any necessary memories about the user. 

Use parallel tool calling to handle updates and insertions simultaneously:"""

def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Load memories from the store and use them to personalize the chatbot's response."""
    
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Retrieve memory from the store
    namespace = ("memories", user_id)
    memories = store.search(namespace)

    # Format the memories for the system prompt
    info = "\n".join(f"- {mem.value['content']}" for mem in memories)
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=info)

    # Respond using memory as well as the chat history
    response = model.invoke([SystemMessage(content=system_msg)]+state["messages"])

    return {"messages": response}

def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the memory collection."""
    
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Define the namespace for the memories
    namespace = ("memories", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "Memory"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None
                        )

    # Merge the chat history and the instruction
    updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION)] + state["messages"]))

    # Invoke the extractor
    result = trustcall_extractor.invoke({"messages": updated_messages, 
                                        "existing": existing_memories})

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )

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


    
![jpeg](memoryschema_collection_files/memoryschema_collection_28_0.jpg)
    



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
    
    Hi Lance! It's great to meet you. How can I assist you today?



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
    
    That sounds like a lot of fun! San Francisco has some beautiful routes for biking. Do you have a favorite trail or area you like to explore?



```python
# Namespace for the memory to save
user_id = "1"
namespace = ("memories", user_id)
memories = across_thread_memory.search(namespace)
for m in memories:
    print(m.dict())
```

    {'value': {'content': "User's name is Lance."}, 'key': 'dee65880-dd7d-4184-8ca1-1f7400f7596b', 'namespace': ['memories', '1'], 'created_at': '2024-10-30T22:18:52.413283+00:00', 'updated_at': '2024-10-30T22:18:52.413284+00:00'}
    {'value': {'content': 'User likes to bike around San Francisco.'}, 'key': '662195fc-8ea4-4f64-a6b6-6b86d9cb85c0', 'namespace': ['memories', '1'], 'created_at': '2024-10-30T22:18:56.597813+00:00', 'updated_at': '2024-10-30T22:18:56.597814+00:00'}



```python
# User input 
input_messages = [HumanMessage(content="I also enjoy going to bakeries")]

# Run the graph
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    I also enjoy going to bakeries
    ==================================[1m Ai Message [0m==================================
    
    Biking and bakeries make a great combination! Do you have a favorite bakery in San Francisco, or are you on the hunt for new ones to try?


Continue the conversation in a new thread.


```python
# We supply a thread ID for short-term (within-thread) memory
# We supply a user ID for long-term (across-thread) memory 
config = {"configurable": {"thread_id": "2", "user_id": "1"}}

# User input 
input_messages = [HumanMessage(content="What bakeries do you recommend for me?")]

# Run the graph
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    What bakeries do you recommend for me?
    ==================================[1m Ai Message [0m==================================
    
    Since you enjoy biking around San Francisco, you might like to check out some of these bakeries that are both delicious and located in areas that are great for a bike ride:
    
    1. **Tartine Bakery** - Located in the Mission District, it's famous for its bread and pastries. The area is vibrant and perfect for a leisurely ride.
    
    2. **Arsicault Bakery** - Known for its incredible croissants, it's in the Richmond District, which offers a nice ride through Golden Gate Park.
    
    3. **B. Patisserie** - Situated in Lower Pacific Heights, this bakery is renowned for its kouign-amann and other French pastries. The neighborhood is charming and bike-friendly.
    
    4. **Mr. Holmes Bakehouse** - Famous for its cruffins, it's located in the Tenderloin, which is a bit more urban but still accessible by bike.
    
    5. **Noe Valley Bakery** - A cozy spot in Noe Valley, perfect for a stop after exploring the hilly streets of the area.
    
    Do any of these sound like a good fit for your next biking adventure?


### LangSmith 

https://smith.langchain.com/public/c87543ec-b426-4a82-a3ab-94d01c01d9f4/r

## Studio

![Screenshot 2024-10-30 at 11.29.25 AM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/6732d0876d3daa19fef993ba_Screenshot%202024-11-11%20at%207.50.21%E2%80%AFPM.png)


```python

```
