[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-2/trim-filter-messages.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239435-lesson-4-trim-and-filter-messages)

# Filtering and trimming messages

## Review

Now, we have a deeper understanding of a few things: 

* How to customize the graph state schema
* How to define custom state reducers
* How to use multiple graph state schemas

## Goals

Now, we can start using these concepts with models in LangGraph!
 
In the next few sessions, we'll build towards a chatbot that has long-term memory.

Because our chatbot will use messages, let's first talk a bit more about advanced ways to work with messages in graph state.


```python
%%capture --no-stderr
%pip install --quiet -U langchain_core langgraph langchain_openai
```


```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
```

We'll use [LangSmith](https://docs.smith.langchain.com/) for [tracing](https://docs.smith.langchain.com/concepts/tracing).

We'll log to a project, `langchain-academy`. 


```python
_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
```

## Messages as state

First, let's define some messages.


```python
from pprint import pprint
from langchain_core.messages import AIMessage, HumanMessage
messages = [AIMessage(f"So you said you were researching ocean mammals?", name="Bot")]
messages.append(HumanMessage(f"Yes, I know about whales. But what others should I learn about?", name="Lance"))

for m in messages:
    m.pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    Name: Bot
    
    So you said you were researching ocean mammals?
    ================================[1m Human Message [0m=================================
    Name: Lance
    
    Yes, I know about whales. But what others should I learn about?


Recall we can pass them to a chat model.


```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")
llm.invoke(messages)
```




    AIMessage(content='Great question, Lance! Ocean mammals are a fascinating group of animals. Here are a few more ocean mammals you might want to learn about:\n\n1. **Dolphins**: These intelligent and social creatures are known for their playful behavior and complex communication skills. There are several species of dolphins, including the bottlenose dolphin and the common dolphin.\n\n2. **Porpoises**: Similar to dolphins but typically smaller and stouter, porpoises are less well-known but equally interesting. The harbor porpoise is one example.\n\n3. **Seals**: These include both true seals (like the harbor seal) and eared seals (which include sea lions and fur seals). They are known for their ability to live both in the water and on land.\n\n4. **Sea Lions**: These are a type of eared seal, easily recognized by their external ear flaps and their ability to "walk" on land using their large flippers.\n\n5. **Walruses**: Known for their distinctive long tusks and whiskers, walruses are social animals that live in Arctic regions.\n\n6. **Manatees and Dugongs**: Often called "sea cows," these gentle herbivores are found in warm coastal waters and rivers. Manatees are found in the Americas and Africa, while dugongs are found in the Indo-Pacific region.\n\n7. **Sea Otters**: Although not exclusively marine, sea otters spend much of their time in the water. They are known for their use of tools to open shellfish.\n\n8. **Polar Bears**: While primarily land animals, polar bears are excellent swimmers and spend a significant amount of time hunting on sea ice.\n\n9. **Sperm Whales**: Known for their large heads and deep diving abilities, sperm whales are the largest of the toothed whales.\n\n10. **Narwhals**: Often called the "unicorns of the sea," these Arctic whales are known for their long, spiral tusk, which is actually an elongated tooth.\n\nEach of these animals has unique adaptations and behaviors that make them fascinating subjects of study. Happy researching!', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 434, 'prompt_tokens': 39, 'total_tokens': 473}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_25624ae3a5', 'finish_reason': 'stop', 'logprobs': None}, id='run-513c189f-66e0-4c3c-bdb8-5d59934d10f9-0', usage_metadata={'input_tokens': 39, 'output_tokens': 434, 'total_tokens': 473})



We can run our chat model in a simple graph with `MessagesState`.


```python
from IPython.display import Image, display
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END

# Node
def chat_model_node(state: MessagesState):
    return {"messages": llm.invoke(state["messages"])}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```


    
![jpeg](trim-filter-messages_files/trim-filter-messages_11_0.jpg)
    



```python
output = graph.invoke({'messages': messages})
for m in output['messages']:
    m.pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    Name: Bot
    
    So you said you were researching ocean mammals?
    ================================[1m Human Message [0m=================================
    Name: Lance
    
    Yes, I know about whales. But what others should I learn about?
    ==================================[1m Ai Message [0m==================================
    
    Absolutely, whales are fascinating! But there are many other ocean mammals worth learning about. Here are a few you might find interesting:
    
    1. **Dolphins**: Highly intelligent and social, dolphins are known for their playful behavior and complex communication. There are many species, including the bottlenose dolphin and the orca (killer whale), which is actually the largest member of the dolphin family.
    
    2. **Seals and Sea Lions**: These pinnipeds are often found lounging on beaches or frolicking in the water. Seals tend to be more solitary, while sea lions are social and known for their barking calls.
    
    3. **Manatees and Dugongs**: Often referred to as sea cows, these gentle herbivores graze on seagrasses in shallow coastal areas. Manatees are found in the Atlantic waters, while dugongs are found in the Indo-Pacific region.
    
    4. **Walruses**: Known for their distinctive tusks, walruses are large, social pinnipeds that inhabit the Arctic region. They use their tusks for various purposes, including pulling themselves out of the water and breaking through ice.
    
    5. **Narwhals**: Sometimes called the "unicorns of the sea," narwhals are known for their long, spiral tusks, which are actually elongated teeth. They live in Arctic waters and are relatively elusive.
    
    6. **Porpoises**: Similar to dolphins but generally smaller and with different physical characteristics, porpoises are also highly intelligent and social animals. They are less acrobatic than dolphins and have more triangular dorsal fins.
    
    7. **Sea Otters**: Found along the coasts of the northern and eastern North Pacific Ocean, sea otters are known for their use of tools and their dense fur, which is the thickest of any animal.
    
    8. **Polar Bears**: Though they spend a lot of time on ice, polar bears are excellent swimmers and are considered marine mammals because they depend on the ocean for their primary food source, seals.
    
    Each of these ocean mammals has unique adaptations and behaviors that make them interesting subjects of study. If you're into marine biology, you might find their various ecosystems, social structures, and survival strategies particularly compelling.


## Reducer

A practical challenge when working with messages is managing long-running conversations. 

Long-running conversations result in high token usage and latency if we are not careful, because we pass a growing list of messages to the model.

We have a few ways to address this.

First, recall the trick we saw using `RemoveMessage` and the `add_messages` reducer.


```python
from langchain_core.messages import RemoveMessage

# Nodes
def filter_messages(state: MessagesState):
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"messages": delete_messages}

def chat_model_node(state: MessagesState):    
    return {"messages": [llm.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("filter", filter_messages)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "filter")
builder.add_edge("filter", "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```


    
![jpeg](trim-filter-messages_files/trim-filter-messages_14_0.jpg)
    



```python
# Message list with a preamble
messages = [AIMessage("Hi.", name="Bot", id="1")]
messages.append(HumanMessage("Hi.", name="Lance", id="2"))
messages.append(AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"))
messages.append(HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Lance", id="4"))

# Invoke
output = graph.invoke({'messages': messages})
for m in output['messages']:
    m.pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    Name: Bot
    
    So you said you were researching ocean mammals?
    ================================[1m Human Message [0m=================================
    Name: Lance
    
    Yes, I know about whales. But what others should I learn about?
    ==================================[1m Ai Message [0m==================================
    
    That's great that you know about whales! There are a variety of other fascinating ocean mammals you might be interested in learning about. Here are a few:
    
    1. **Dolphins**: These highly intelligent and social animals are part of the cetacean family, which also includes whales and porpoises. There are many species of dolphins, including the common bottlenose dolphin and the orca, or killer whale, which is actually the largest member of the dolphin family.
    
    2. **Porpoises**: Similar to dolphins but generally smaller and with different facial structures and teeth. The harbor porpoise is one of the more well-known species.
    
    3. **Seals and Sea Lions**: These pinnipeds are known for their playful nature and agility in water. Seals typically have smaller flippers and no visible ear flaps, while sea lions have larger flippers and visible ear flaps.
    
    4. **Walruses**: Recognizable by their large tusks, whiskers, and significant bulk, walruses are pinnipeds as well and are usually found in Arctic regions.
    
    5. **Manatees and Dugongs**: These gentle giants, often called sea cows, are slow-moving and primarily herbivorous. Manatees are found in the Caribbean and the Gulf of Mexico, while dugongs inhabit the coastal waters of the Indian and western Pacific Oceans.
    
    6. **Sea Otters**: Known for their use of tools to open shells and their thick fur, sea otters are a keystone species in their ecosystems, particularly in kelp forest habitats along the Pacific coast of North America.
    
    7. **Polar Bears**: While not exclusively marine, polar bears depend heavily on the ocean for hunting seals and are excellent swimmers.
    
    Each of these groups has unique adaptations and behaviors that make them fascinating subjects of study. Happy researching!


## Filtering messages

If you don't need or want to modify the graph state, you can just filter the messages you pass to the chat model.

For example, just pass in a filtered list: `llm.invoke(messages[-1:])` to the model.


```python
# Node
def chat_model_node(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"][-1:])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```


    
![jpeg](trim-filter-messages_files/trim-filter-messages_17_0.jpg)
    


Let's take our existing list of messages, append the above LLM response, and append a follow-up question.


```python
messages.append(output['messages'][-1])
messages.append(HumanMessage(f"Tell me more about Narwhals!", name="Lance"))
```


```python
for m in messages:
    m.pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    Name: Bot
    
    Hi.
    ================================[1m Human Message [0m=================================
    Name: Lance
    
    Hi.
    ==================================[1m Ai Message [0m==================================
    Name: Bot
    
    So you said you were researching ocean mammals?
    ================================[1m Human Message [0m=================================
    Name: Lance
    
    Yes, I know about whales. But what others should I learn about?
    ==================================[1m Ai Message [0m==================================
    
    That's great that you know about whales! There are many other fascinating ocean mammals you can learn about. Here are a few:
    
    1. **Dolphins**: Highly intelligent and social animals, dolphins are known for their playful behavior and sophisticated communication skills. There are many species of dolphins, including the well-known bottlenose dolphin.
    
    2. **Porpoises**: Often confused with dolphins, porpoises are smaller and have different body shapes and teeth. They are generally more reclusive and less acrobatic than dolphins.
    
    3. **Seals**: Seals are part of the pinniped family, which also includes sea lions and walruses. They have streamlined bodies and flippers, making them excellent swimmers. Common types of seals include harbor seals and elephant seals.
    
    4. **Sea Lions**: Similar to seals but with some key differences, sea lions have external ear flaps and can rotate their hind flippers to walk on land. They are also very social and often gather in large groups.
    
    5. **Walruses**: Recognizable by their long tusks and whiskers, walruses are large marine mammals that are found in Arctic regions. They use their tusks to help them climb out of the water and to break through ice.
    
    6. **Manatees and Dugongs**: These gentle giants are often referred to as sea cows. Manatees are found in the Atlantic Ocean, while dugongs are found in the Indian and Pacific Oceans. They are herbivores and spend most of their time grazing on underwater vegetation.
    
    7. **Sea Otters**: Known for their playful behavior and use of tools, sea otters are an important part of the marine ecosystem. They have thick fur to keep them warm in cold waters and are often seen floating on their backs.
    
    8. **Polar Bears**: While not exclusively marine, polar bears spend a significant amount of time in the ocean, particularly in Arctic regions. They are excellent swimmers and rely on sea ice to hunt seals, their primary food source.
    
    9. **Narwhals**: Often called the "unicorns of the sea," narwhals have a long, spiral tusk that is actually an elongated tooth. They are found in Arctic waters and are known for their deep diving abilities.
    
    10. **Orcas (Killer Whales)**: Though they are technically a type of dolphin, orcas are often considered separately due to their size and distinctive black-and-white coloring. They are apex predators and have complex social structures.
    
    Each of these ocean mammals has unique behaviors, adaptations, and ecological roles, making them fascinating subjects for study.
    ================================[1m Human Message [0m=================================
    Name: Lance
    
    Tell me more about Narwhals!



```python
# Invoke, using message filtering
output = graph.invoke({'messages': messages})
for m in output['messages']:
    m.pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    Name: Bot
    
    Hi.
    ================================[1m Human Message [0m=================================
    Name: Lance
    
    Hi.
    ==================================[1m Ai Message [0m==================================
    Name: Bot
    
    So you said you were researching ocean mammals?
    ================================[1m Human Message [0m=================================
    Name: Lance
    
    Yes, I know about whales. But what others should I learn about?
    ==================================[1m Ai Message [0m==================================
    
    That's great that you know about whales! There are many other fascinating ocean mammals you can learn about. Here are a few:
    
    1. **Dolphins**: Highly intelligent and social animals, dolphins are known for their playful behavior and sophisticated communication skills. There are many species of dolphins, including the well-known bottlenose dolphin.
    
    2. **Porpoises**: Often confused with dolphins, porpoises are smaller and have different body shapes and teeth. They are generally more reclusive and less acrobatic than dolphins.
    
    3. **Seals**: Seals are part of the pinniped family, which also includes sea lions and walruses. They have streamlined bodies and flippers, making them excellent swimmers. Common types of seals include harbor seals and elephant seals.
    
    4. **Sea Lions**: Similar to seals but with some key differences, sea lions have external ear flaps and can rotate their hind flippers to walk on land. They are also very social and often gather in large groups.
    
    5. **Walruses**: Recognizable by their long tusks and whiskers, walruses are large marine mammals that are found in Arctic regions. They use their tusks to help them climb out of the water and to break through ice.
    
    6. **Manatees and Dugongs**: These gentle giants are often referred to as sea cows. Manatees are found in the Atlantic Ocean, while dugongs are found in the Indian and Pacific Oceans. They are herbivores and spend most of their time grazing on underwater vegetation.
    
    7. **Sea Otters**: Known for their playful behavior and use of tools, sea otters are an important part of the marine ecosystem. They have thick fur to keep them warm in cold waters and are often seen floating on their backs.
    
    8. **Polar Bears**: While not exclusively marine, polar bears spend a significant amount of time in the ocean, particularly in Arctic regions. They are excellent swimmers and rely on sea ice to hunt seals, their primary food source.
    
    9. **Narwhals**: Often called the "unicorns of the sea," narwhals have a long, spiral tusk that is actually an elongated tooth. They are found in Arctic waters and are known for their deep diving abilities.
    
    10. **Orcas (Killer Whales)**: Though they are technically a type of dolphin, orcas are often considered separately due to their size and distinctive black-and-white coloring. They are apex predators and have complex social structures.
    
    Each of these ocean mammals has unique behaviors, adaptations, and ecological roles, making them fascinating subjects for study.
    ================================[1m Human Message [0m=================================
    Name: Lance
    
    Tell me more about Narwhals!
    ==================================[1m Ai Message [0m==================================
    
    Of course! Narwhals (Monodon monoceros) are fascinating marine mammals that belong to the family Monodontidae, which also includes the beluga whale. They are best known for the long, spiral tusk that protrudes from the head of the males, which has earned them the nickname "unicorns of the sea."
    
    Here are some key facts about narwhals:
    
    ### Physical Characteristics
    - **Tusk**: The most distinctive feature of the narwhal is the tusk, which is actually an elongated tooth. It can grow up to 10 feet (3 meters) long and is usually found in males, though some females may also develop smaller tusks. The tusk grows in a spiral pattern and is thought to have sensory capabilities, with millions of nerve endings.
    - **Body**: Narwhals have a stocky body with a mottled black and white skin pattern. They lack a dorsal fin, which is thought to be an adaptation to swimming under ice.
    - **Size**: Adult narwhals typically range from 13 to 20 feet (4 to 6 meters) in length, with males generally being larger than females.
    - **Weight**: They can weigh between 1,760 to 3,530 pounds (800 to 1,600 kilograms).
    
    ### Habitat and Distribution
    - Narwhals are native to the Arctic waters of Canada, Greenland, Norway, and Russia. They are especially common in the Baffin Bay and the waters surrounding Greenland.
    - They prefer deep waters and are often found in areas with heavy sea ice.
    
    ### Behavior and Diet
    - **Diving**: Narwhals are deep divers and can reach depths of up to 5,000 feet (1,500 meters) in search of food. They can hold their breath for up to 25 minutes.
    - **Diet**: Their diet primarily consists of fish such as Arctic cod and Greenland halibut, as well as squid and shrimp.
    - **Social Structure**: Narwhals are social animals and are often found in groups called pods, which typically consist of 5 to 10 individuals but can sometimes number in the hundreds.
    
    ### Reproduction and Lifespan
    - Females give birth to a single calf after a gestation period of about 14 to 15 months. Calves are usually born in the spring or early summer.
    - Narwhals have a long lifespan and can live up to 50 years, although some individuals may live even longer.
    
    ### Conservation Status
    - Narwhals are currently classified as "Near Threatened" by the International Union for Conservation of Nature (IUCN). Their main threats include climate change, which affects their sea ice habitat, and human activities such as shipping and oil exploration.
    - Indigenous communities in the Arctic have traditionally hunted narwhals for their meat, blubber, and tusks, which are used for various purposes, including art and tools.
    
    ### Cultural Significance
    - The narwhal's tusk has fascinated humans for centuries and was often sold as a "unicorn horn" in medieval Europe, believed to possess magical properties.
    - Narwhals hold significant cultural and economic value for indigenous Arctic communities.
    
    Overall, narwhals are remarkable creatures with unique adaptations that allow them to thrive in some of the planet's harshest environments.


The state has all of the mesages.

But, let's look at the LangSmith trace to see that the model invocation only uses the last message:

https://smith.langchain.com/public/75aca3ce-ef19-4b92-94be-0178c7a660d9/r

## Trim messages

Another approach is to [trim messages](https://python.langchain.com/v0.2/docs/how_to/trim_messages/#getting-the-last-max_tokens-tokens), based upon a set number of tokens. 

This restricts the message history to a specified number of tokens.

While filtering only returns a post-hoc subset of the messages between agents, trimming restricts the number of tokens that a chat model can use to respond.

See the `trim_messages` below.


```python
from langchain_core.messages import trim_messages

# Node
def chat_model_node(state: MessagesState):
    messages = trim_messages(
            state["messages"],
            max_tokens=100,
            strategy="last",
            token_counter=ChatOpenAI(model="gpt-4o"),
            allow_partial=False,
        )
    return {"messages": [llm.invoke(messages)]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```


    
![jpeg](trim-filter-messages_files/trim-filter-messages_24_0.jpg)
    



```python
messages.append(output['messages'][-1])
messages.append(HumanMessage(f"Tell me where Orcas live!", name="Lance"))
```


```python
# Example of trimming messages
trim_messages(
            messages,
            max_tokens=100,
            strategy="last",
            token_counter=ChatOpenAI(model="gpt-4o"),
            allow_partial=False
        )
```




    [HumanMessage(content='Tell me where Orcas live!', name='Lance')]




```python
# Invoke, using message trimming in the chat_model_node 
messages_out_trim = graph.invoke({'messages': messages})
```

Let's look at the LangSmith trace to see the model invocation:

https://smith.langchain.com/public/b153f7e9-f1a5-4d60-8074-f0d7ab5b42ef/r
