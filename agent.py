from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from llm import llm 
from langchain.tools import Tool    
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from tools.vector import kg_qa
from tools.cypher import cypher_qa
from tools.docqa import doc_qa

# Include the LLM from a previous lesson
from llm import llm
# tag::tools[]
#,
   # Tool.from_function(
   #     name="Custom Doc Vector Search Index",
   #     description="Provides information about moive fact details",
   #     func = doc_qa,
   #     return_direct=False
   # )
tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat not covered by other tools",
        func=llm.invoke,
        return_direct=False
    ),
    Tool.from_function(
        name="Cypher QA",
        description="Provide information about movies questions using Cypher",
        func = cypher_qa,
        return_direct=False
    ),
    Tool.from_function(
        name="Vector Search Index",
        description="Provides information about movie plots using Vector Search",
        func = kg_qa,
        return_direct=False
    )
]
# end::tools[]
  

memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)
#Do not answer any questions that do not relate to movies, actors or directors.
#agent_prompt = hub.pull("hwchase17/react-chat")
agent_prompt = PromptTemplate.from_template("""
You are a movie expert providing information about movies.
Be as helpful as possible and return as much information as possible.
Always respond with something.
                                            
TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action must be one of the {tools}
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]                                         
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    handle_parsing_errors=True,
    max_iterations = 3,
    verbose=True
    )




# tag::generate_response[]
def generate_response(prompt):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = agent_executor.invoke({"input": prompt})

    return response['output']
# end::generate_response[]

