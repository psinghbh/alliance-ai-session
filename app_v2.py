from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor

# 1. SETUP - Load Environment and Initialize Model
# -----------------------------------------------------------------------
load_dotenv()
#print(os.environ.get("OPENAI_API_KEY"))
llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")

print("--- AI Agent Session Initiated for Demo ---")

# 2. DEFINE THE CUSTOM TOOL FUNCTION
# -----------------------------------------------------------------------
@tool
def getLeaveDuration(message: str) -> str:
    """
    Analyzes a slack message and determines the duration of the leave mentioned.
    Returns the duration as a simple string (e.g., '1 day', '0.5 day', '5 days').
    """
    # NOTE: In a real application, this logic would use regex, date parsing, 
    # or another LLM call with a Pydantic structure for reliable extraction.
    
    message_lower = message.lower()
    # Logic to extract duration based on keywords
    if "first half" in message_lower or "firsthalf" in message_lower:
        return "0.5 day"
    elif "today and tomorrow" in message_lower:
        return "2 days"
    elif "from" in message_lower and "to" in message_lower:
        # Simple, non-robust date range calculation for demo purposes:
        return "5 days (estimated date range)"
    elif "tomorrow" in message_lower:
        return "1 day"
    elif "ooo today" in message_lower or "sick today" in message_lower or "off today" in message_lower:
        return "1 day"
    else:
        return "Duration is ambiguous or not stated clearly."
        # NOTE: In real application, this can also be handled by other ways
        # such as sending message to LLM, and check if they can parse the message

# 3. DEFINE THE AGENT PROMPT
# -----------------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert HR and AI assistant. When a user provides a slack message, use the 'getLeaveDuration' tool to extract the leave duration. Only answer the user's question."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# 4. CREATE TOOL LIST AND AGENT
# -----------------------------------------------------------------------
tools = [getLeaveDuration] # The list of tools the agent can use

# ----------------------------------------------------------------------
# THIS IS THE CORE LCEL IMPLEMENTATION:
# The `create_openai_tools_agent` automatically inserts the tool-calling logic 
# between the prompt and the LLM, but we express the core flow as a simple chain.
# ----------------------------------------------------------------------
agent = create_openai_tools_agent(llm, tools, prompt)

# 3c. Create the Agent Executor (The final execution loop)
# Set verbose=True to clearly show the "Thought -> Invoking Tool -> Observation -> Final Answer" loop.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. RUN THE AGENT WITH SAMPLE MESSAGES
# -----------------------------------------------------------------------
sample_messages = [
    "Mild fever and throat infection, out sick today.", 
    "Will be OOO from 2nd jan to 7 jan. I need the duration.", 
    "OOO for the first half - doctors appointment. What is the leave duration?",
    "Tell me a joke about cats." # Test case where the tool should NOT be used
]

# 6. SEND PROMPT TO LLM AND GET RESPONSE
# -----------------------------------------------------------------------
for msg in sample_messages:
    print(f"\n--- USER QUERY: {msg} ---")
    response = agent_executor.invoke({"input": msg})
    print(f"AGENT FINAL ANSWER: {response['output']}")

