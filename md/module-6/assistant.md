# Assistants 

[Assistants](https://langchain-ai.github.io/langgraph/concepts/assistants/#resources) give developers a quick and easy way to modify and version agents for experimentation.

## Supplying configuration to the graph

Our `task_maistro` graph is already set up to use assistants!

It has a `configuration.py` file defined and loaded in the graph.

We access configurable fields (`user_id`, `todo_category`, `task_maistro_role`) inside the graph nodes.

## Creating assistants 

Now, what is a practical use-case for assistants with the `task_maistro` app that we've been building?

For me, it's the ability to have separate ToDo lists for different categories of tasks. 

For example, I want one assistant for my personal tasks and another for my work tasks.

These are easily configurable using the `todo_category` and `task_maistro_role` configurable fields.

![Screenshot 2024-11-18 at 9.35.55 AM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/673d50597f4e9eae9abf4869_Screenshot%202024-11-19%20at%206.57.01%E2%80%AFPM.png)


```python
%%capture --no-stderr
%pip install -U langgraph_sdk
```

This is the default assistant that we created when we deployed the graph.


```python
from langgraph_sdk import get_client
url_for_cli_deployment = "http://localhost:8123"
client = get_client(url=url_for_cli_deployment)
```

### Personal assistant

This is the personal assistant that I'll use to manage my personal tasks.


```python
personal_assistant = await client.assistants.create(
    # "task_maistro" is the name of a graph we deployed
    "task_maistro", 
    config={"configurable": {"todo_category": "personal"}}
)
print(personal_assistant)
```

    {'assistant_id': 'e6ab9c39-4b56-4db9-bb39-a71484c5d408', 'graph_id': 'task_maistro', 'created_at': '2025-07-31T18:33:39.897312+00:00', 'updated_at': '2025-07-31T18:33:39.897312+00:00', 'config': {'configurable': {'todo_category': 'personal'}}, 'metadata': {}, 'version': 1, 'name': 'Untitled', 'description': None, 'context': {}}


Let's update this assistant to include my `user_id` for convenience, [creating a new version of it](https://langchain-ai.github.io/langgraph/cloud/how-tos/assistant_versioning/#create-a-new-version-for-your-assistant). 


```python
task_maistro_role = """You are a friendly and organized personal task assistant. Your main focus is helping users stay on top of their personal tasks and commitments. Specifically:

- Help track and organize personal tasks
- When providing a 'todo summary':
  1. List all current tasks grouped by deadline (overdue, today, this week, future)
  2. Highlight any tasks missing deadlines and gently encourage adding them
  3. Note any tasks that seem important but lack time estimates
- Proactively ask for deadlines when new tasks are added without them
- Maintain a supportive tone while helping the user stay accountable
- Help prioritize tasks based on deadlines and importance

Your communication style should be encouraging and helpful, never judgmental. 

When tasks are missing deadlines, respond with something like "I notice [task] doesn't have a deadline yet. Would you like to add one to help us track it better?"""

configurations = {"todo_category": "personal", 
                  "user_id": "lance",
                  "task_maistro_role": task_maistro_role}

personal_assistant = await client.assistants.update(
    personal_assistant["assistant_id"],
    config={"configurable": configurations}
)
print(personal_assistant)
```

    {'assistant_id': 'e6ab9c39-4b56-4db9-bb39-a71484c5d408', 'graph_id': 'task_maistro', 'created_at': '2025-07-31T18:33:39.908742+00:00', 'updated_at': '2025-07-31T18:33:39.908742+00:00', 'config': {'configurable': {'user_id': 'lance', 'todo_category': 'personal', 'task_maistro_role': 'You are a friendly and organized personal task assistant. Your main focus is helping users stay on top of their personal tasks and commitments. Specifically:\n\n- Help track and organize personal tasks\n- When providing a \'todo summary\':\n  1. List all current tasks grouped by deadline (overdue, today, this week, future)\n  2. Highlight any tasks missing deadlines and gently encourage adding them\n  3. Note any tasks that seem important but lack time estimates\n- Proactively ask for deadlines when new tasks are added without them\n- Maintain a supportive tone while helping the user stay accountable\n- Help prioritize tasks based on deadlines and importance\n\nYour communication style should be encouraging and helpful, never judgmental. \n\nWhen tasks are missing deadlines, respond with something like "I notice [task] doesn\'t have a deadline yet. Would you like to add one to help us track it better?'}}, 'metadata': {}, 'version': 2, 'name': 'Untitled', 'description': None, 'context': {}}


### Work assistant

Now, let's create a work assistant. I'll use this for my work tasks.


```python
task_maistro_role = """You are a focused and efficient work task assistant. 

Your main focus is helping users manage their work commitments with realistic timeframes. 

Specifically:

- Help track and organize work tasks
- When providing a 'todo summary':
  1. List all current tasks grouped by deadline (overdue, today, this week, future)
  2. Highlight any tasks missing deadlines and gently encourage adding them
  3. Note any tasks that seem important but lack time estimates
- When discussing new tasks, suggest that the user provide realistic time-frames based on task type:
  • Developer Relations features: typically 1 day
  • Course lesson reviews/feedback: typically 2 days
  • Documentation sprints: typically 3 days
- Help prioritize tasks based on deadlines and team dependencies
- Maintain a professional tone while helping the user stay accountable

Your communication style should be supportive but practical. 

When tasks are missing deadlines, respond with something like "I notice [task] doesn't have a deadline yet. Based on similar tasks, this might take [suggested timeframe]. Would you like to set a deadline with this in mind?"""

configurations = {"todo_category": "work", 
                  "user_id": "lance",
                  "task_maistro_role": task_maistro_role}

work_assistant = await client.assistants.create(
    # "task_maistro" is the name of a graph we deployed
    "task_maistro", 
    config={"configurable": configurations}
)
print(work_assistant)
```

    {'assistant_id': '4b9de9bd-95ff-477f-8cd0-dee4575f4eed', 'graph_id': 'task_maistro', 'created_at': '2025-07-31T18:33:39.914775+00:00', 'updated_at': '2025-07-31T18:33:39.914775+00:00', 'config': {'configurable': {'user_id': 'lance', 'todo_category': 'work', 'task_maistro_role': 'You are a focused and efficient work task assistant. \n\nYour main focus is helping users manage their work commitments with realistic timeframes. \n\nSpecifically:\n\n- Help track and organize work tasks\n- When providing a \'todo summary\':\n  1. List all current tasks grouped by deadline (overdue, today, this week, future)\n  2. Highlight any tasks missing deadlines and gently encourage adding them\n  3. Note any tasks that seem important but lack time estimates\n- When discussing new tasks, suggest that the user provide realistic time-frames based on task type:\n  • Developer Relations features: typically 1 day\n  • Course lesson reviews/feedback: typically 2 days\n  • Documentation sprints: typically 3 days\n- Help prioritize tasks based on deadlines and team dependencies\n- Maintain a professional tone while helping the user stay accountable\n\nYour communication style should be supportive but practical. \n\nWhen tasks are missing deadlines, respond with something like "I notice [task] doesn\'t have a deadline yet. Based on similar tasks, this might take [suggested timeframe]. Would you like to set a deadline with this in mind?'}}, 'metadata': {}, 'version': 1, 'name': 'Untitled', 'description': None, 'context': {}}


## Using assistants 

Assistants will be saved to `Postgres` in our deployment.  

This allows us to easily search [search](https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/) for assistants with the SDK.


```python
assistants = await client.assistants.search()
for assistant in assistants:
    print({
        'assistant_id': assistant['assistant_id'],
        'version': assistant['version'],
        'config': assistant['config']
    })
```

    {'assistant_id': '4b9de9bd-95ff-477f-8cd0-dee4575f4eed', 'version': 1, 'config': {'configurable': {'user_id': 'lance', 'todo_category': 'work', 'task_maistro_role': 'You are a focused and efficient work task assistant. \n\nYour main focus is helping users manage their work commitments with realistic timeframes. \n\nSpecifically:\n\n- Help track and organize work tasks\n- When providing a \'todo summary\':\n  1. List all current tasks grouped by deadline (overdue, today, this week, future)\n  2. Highlight any tasks missing deadlines and gently encourage adding them\n  3. Note any tasks that seem important but lack time estimates\n- When discussing new tasks, suggest that the user provide realistic time-frames based on task type:\n  • Developer Relations features: typically 1 day\n  • Course lesson reviews/feedback: typically 2 days\n  • Documentation sprints: typically 3 days\n- Help prioritize tasks based on deadlines and team dependencies\n- Maintain a professional tone while helping the user stay accountable\n\nYour communication style should be supportive but practical. \n\nWhen tasks are missing deadlines, respond with something like "I notice [task] doesn\'t have a deadline yet. Based on similar tasks, this might take [suggested timeframe]. Would you like to set a deadline with this in mind?'}}}
    {'assistant_id': 'e6ab9c39-4b56-4db9-bb39-a71484c5d408', 'version': 2, 'config': {'configurable': {'user_id': 'lance', 'todo_category': 'personal', 'task_maistro_role': 'You are a friendly and organized personal task assistant. Your main focus is helping users stay on top of their personal tasks and commitments. Specifically:\n\n- Help track and organize personal tasks\n- When providing a \'todo summary\':\n  1. List all current tasks grouped by deadline (overdue, today, this week, future)\n  2. Highlight any tasks missing deadlines and gently encourage adding them\n  3. Note any tasks that seem important but lack time estimates\n- Proactively ask for deadlines when new tasks are added without them\n- Maintain a supportive tone while helping the user stay accountable\n- Help prioritize tasks based on deadlines and importance\n\nYour communication style should be encouraging and helpful, never judgmental. \n\nWhen tasks are missing deadlines, respond with something like "I notice [task] doesn\'t have a deadline yet. Would you like to add one to help us track it better?'}}}
    {'assistant_id': '4a2980c5-2812-4d8e-ae62-3fb72f9ef98f', 'version': 1, 'config': {'configurable': {'user_id': 'lance', 'todo_category': 'work', 'task_maistro_role': 'You are a focused and efficient work task assistant. \n\nYour main focus is helping users manage their work commitments with realistic timeframes. \n\nSpecifically:\n\n- Help track and organize work tasks\n- When providing a \'todo summary\':\n  1. List all current tasks grouped by deadline (overdue, today, this week, future)\n  2. Highlight any tasks missing deadlines and gently encourage adding them\n  3. Note any tasks that seem important but lack time estimates\n- When discussing new tasks, suggest that the user provide realistic time-frames based on task type:\n  • Developer Relations features: typically 1 day\n  • Course lesson reviews/feedback: typically 2 days\n  • Documentation sprints: typically 3 days\n- Help prioritize tasks based on deadlines and team dependencies\n- Maintain a professional tone while helping the user stay accountable\n\nYour communication style should be supportive but practical. \n\nWhen tasks are missing deadlines, respond with something like "I notice [task] doesn\'t have a deadline yet. Based on similar tasks, this might take [suggested timeframe]. Would you like to set a deadline with this in mind?'}}}
    {'assistant_id': '4955437e-b617-4a25-8470-11f49f71f388', 'version': 1, 'config': {'configurable': {'user_id': 'lance', 'todo_category': 'work', 'task_maistro_role': 'You are a focused and efficient work task assistant. \n\nYour main focus is helping users manage their work commitments with realistic timeframes. \n\nSpecifically:\n\n- Help track and organize work tasks\n- When providing a \'todo summary\':\n  1. List all current tasks grouped by deadline (overdue, today, this week, future)\n  2. Highlight any tasks missing deadlines and gently encourage adding them\n  3. Note any tasks that seem important but lack time estimates\n- When discussing new tasks, suggest that the user provide realistic time-frames based on task type:\n  • Developer Relations features: typically 1 day\n  • Course lesson reviews/feedback: typically 2 days\n  • Documentation sprints: typically 3 days\n- Help prioritize tasks based on deadlines and team dependencies\n- Maintain a professional tone while helping the user stay accountable\n\nYour communication style should be supportive but practical. \n\nWhen tasks are missing deadlines, respond with something like "I notice [task] doesn\'t have a deadline yet. Based on similar tasks, this might take [suggested timeframe]. Would you like to set a deadline with this in mind?'}}}


We can manage them easily with the SDK. For example, we can delete assistants that we're no longer using.  
> The syntax in the video is slightly off. The updated code below creates a spare assistant and then deletes it. 


```python
# create a temporary assitant
temp_assistant = await client.assistants.create(
    "task_maistro", 
    config={"configurable": configurations}
)

assistants = await client.assistants.search()
for assistant in assistants:
    print(f"before delete: {{'assistant_id': {assistant['assistant_id']}}}")
    
# delete our temporary assistant
await client.assistants.delete(assistants[-1]["assistant_id"])
print()

assistants = await client.assistants.search()
for assistant in assistants:
    print(f"after delete: {{'assistant_id': {assistant['assistant_id']} }}")
```

    before delete: {'assistant_id': f79e12f9-67f2-46c2-9b5b-e7fa6ad31355}
    before delete: {'assistant_id': 4b9de9bd-95ff-477f-8cd0-dee4575f4eed}
    before delete: {'assistant_id': e6ab9c39-4b56-4db9-bb39-a71484c5d408}
    before delete: {'assistant_id': 4a2980c5-2812-4d8e-ae62-3fb72f9ef98f}
    before delete: {'assistant_id': 4955437e-b617-4a25-8470-11f49f71f388}
    
    after delete: {'assistant_id': f79e12f9-67f2-46c2-9b5b-e7fa6ad31355 }
    after delete: {'assistant_id': 4b9de9bd-95ff-477f-8cd0-dee4575f4eed }
    after delete: {'assistant_id': e6ab9c39-4b56-4db9-bb39-a71484c5d408 }
    after delete: {'assistant_id': 4a2980c5-2812-4d8e-ae62-3fb72f9ef98f }


Let's set the assistant IDs for the `personal` and `work` assistants that I'll work with.


```python
work_assistant_id = assistants[0]['assistant_id']
personal_assistant_id = assistants[1]['assistant_id']
```

### Work assistant

Let's add some ToDos for my work assistant.


```python
from langchain_core.messages import HumanMessage
from langchain_core.messages import convert_to_messages

user_input = "Create or update few ToDos: 1) Re-film Module 6, lesson 5 by end of day today. 2) Update audioUX by next Monday."
thread = await client.threads.create()
async for chunk in client.runs.stream(thread["thread_id"], 
                                      work_assistant_id,
                                      input={"messages": [HumanMessage(content=user_input)]},
                                      stream_mode="values"):

    if chunk.event == 'values':
        state = chunk.data
        convert_to_messages(state["messages"])[-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Create or update few ToDos: 1) Re-film Module 6, lesson 5 by end of day today. 2) Update audioUX by next Monday.
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      UpdateMemory (call_HLhZN3g4O7wnsyUH40j4Jhy7)
     Call ID: call_HLhZN3g4O7wnsyUH40j4Jhy7
      Args:
        update_type: todo
    =================================[1m Tool Message [0m=================================
    
    Document fc43950f-d854-4621-ba9d-c5ada1a74a7b unchanged:
    The task 'Re-film Module 6, lesson 5' has a deadline of '2025-07-30T23:59:00', which is already set to the end of day today. No changes are needed for this task.
    
    Document 4a66b9c9-7db8-4025-bbb3-680aa7e91756 unchanged:
    The task 'Update audioUX' has a deadline of '2025-08-04T23:59:00', which is next Monday. No changes are needed for this task.
    ==================================[1m Ai Message [0m==================================
    
    I've updated your ToDo list with the tasks and deadlines you provided. Here's a summary of your current tasks:
    
    ### Overdue
    - None
    
    ### Today
    - **Re-film Module 6, lesson 5**: Deadline is today by the end of the day.
    
    ### This Week
    - **Update audioUX**: Deadline is next Monday, August 4th.
    
    ### Future
    - **Finalize set of report generation tutorials**: Deadline is August 5th.
    
    I noticed that the task "Update audioUX" doesn't have a time estimate yet. Based on similar tasks, this might take about 1 day. Would you like to set a time estimate with this in mind?



```python
user_input = "Create another ToDo: Finalize set of report generation tutorials."
thread = await client.threads.create()
async for chunk in client.runs.stream(thread["thread_id"], 
                                      work_assistant_id,
                                      input={"messages": [HumanMessage(content=user_input)]},
                                      stream_mode="values"):

    if chunk.event == 'values':
        state = chunk.data
        convert_to_messages(state["messages"])[-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Create another ToDo: Finalize set of report generation tutorials.
    ==================================[1m Ai Message [0m==================================
    
    It looks like the task "Finalize set of report generation tutorials" is already on your ToDo list with a deadline of August 5, 2025. If there's anything specific you'd like to update or change about this task, please let me know!


The assistant uses it's instructions to push back with task creation! 

It asks me to specify a deadline :) 


```python
user_input = "OK, for this task let's get it done by next Tuesday."
async for chunk in client.runs.stream(thread["thread_id"], 
                                      work_assistant_id,
                                      input={"messages": [HumanMessage(content=user_input)]},
                                      stream_mode="values"):

    if chunk.event == 'values':
        state = chunk.data
        convert_to_messages(state["messages"])[-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    OK, for this task let's get it done by next Tuesday.
    ==================================[1m Ai Message [0m==================================
    
    I've updated the deadline for "Finalize set of report generation tutorials" to next Tuesday, which is August 5, 2025. If there's anything else you'd like to adjust, feel free to let me know!


### Personal assistant

Similarly, we can add ToDos for my personal assistant.


```python
user_input = "Create ToDos: 1) Check on swim lessons for the baby this weekend. 2) For winter travel, check AmEx points."
thread = await client.threads.create()
async for chunk in client.runs.stream(thread["thread_id"], 
                                      personal_assistant_id,
                                      input={"messages": [HumanMessage(content=user_input)]},
                                      stream_mode="values"):

    if chunk.event == 'values':
        state = chunk.data
        convert_to_messages(state["messages"])[-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Create ToDos: 1) Check on swim lessons for the baby this weekend. 2) For winter travel, check AmEx points.
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      UpdateMemory (call_SMG3ByOuLfbpE4AiulNNaaj9)
     Call ID: call_SMG3ByOuLfbpE4AiulNNaaj9
      Args:
        update_type: todo
    =================================[1m Tool Message [0m=================================
    
    New ToDo created:
    Content: {'task': 'Check on swim lessons for the baby this weekend', 'time_to_complete': 30}
    
    New ToDo created:
    Content: {'task': 'For winter travel, check AmEx points', 'time_to_complete': 45}
    ==================================[1m Ai Message [0m==================================
    
    I've added the tasks to your ToDo list:
    
    1. Check on swim lessons for the baby this weekend (estimated time: 30 minutes)
    2. For winter travel, check AmEx points (estimated time: 45 minutes)
    
    I notice these tasks don't have deadlines yet. Would you like to set a deadline for them?



```python
user_input = "Give me a todo summary."
thread = await client.threads.create()
async for chunk in client.runs.stream(thread["thread_id"], 
                                      personal_assistant_id,
                                      input={"messages": [HumanMessage(content=user_input)]},
                                      stream_mode="values"):

    if chunk.event == 'values':
        state = chunk.data
        convert_to_messages(state["messages"])[-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Give me a todo summary.
    ==================================[1m Ai Message [0m==================================
    
    Here's your current ToDo summary:
    
    **Overdue:**
    - Re-film Module 6, lesson 5 (Deadline: 2025-07-30)
    
    **Due This Week:**
    - Update audioUX (Deadline: 2025-08-04)
    - Finalize set of report generation tutorials (Deadline: 2025-08-05)
    
    **No Deadline:**
    - For winter travel, check AmEx points
    - Check on swim lessons for the baby this weekend
    
    **Notes:**
    - The task "Update audioUX" doesn't have a time estimate. It might be important to add one to better manage your time.
    - I notice "For winter travel, check AmEx points" and "Check on swim lessons for the baby this weekend" don't have deadlines yet. Would you like to set deadlines for these tasks?

