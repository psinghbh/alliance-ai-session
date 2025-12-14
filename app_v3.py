from dotenv import load_dotenv
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
# --- REQUIRED FIXES ---
from pydantic import BaseModel, Field # Pydantic V2 for structured output
from langchain.output_parsers import PydanticOutputParser # <-- THIS LINE WAS LIKELY MISSING OR COMMENTED
# ----------------------

# 1. SETUP - Load Environment and Initialize Model
# -----------------------------------------------------------------------
load_dotenv()
llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")

print("--- AI Agent Session Initiated: Advanced Parsing Tool ---")

# 2. PYDANTIC SCHEMA: Define the structured output the LLM must follow
# -----------------------------------------------------------------------
class LeaveInfo(BaseModel):
    """Schema for extracting leave duration and reason from a Slack message."""
    
    duration: str = Field(
        description="The total length of the leave (e.g., '1 day', '0.5 day', '5 days'). Calculate inclusive days for date ranges."
    )
    reason_category: str = Field(
        description="The primary reason for the leave, categorized as 'Sick/Health', 'Personal/Admin', or 'Travel/Holiday'."
    )

# 3. DEFINE THE ADVANCED CUSTOM TOOL FUNCTION
# -----------------------------------------------------------------------
@tool
def getLeaveDuration(message: str) -> str:
    """
    Analyzes a slack message and determines the duration and reason using a specialized 
    internal LLM chain for robust and structured extraction.
    """
    
    # Initialize the parser with the desired schema
    parser = PydanticOutputParser(pydantic_object=LeaveInfo)
    
    # 3a. Create the specialized prompt for extraction (remains the same)
    extraction_template = """
    You are a meticulous data extraction system. Analyze the message below and calculate 
    the duration of leave and the reason category. Output the result ONLY in the 
    required JSON format.

    Message to analyze: "{message}"

    {format_instructions}
    """

    extraction_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", extraction_template),
            ("human", "{message}")
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    
    # 3b. Create the tool's internal LCEL chain (remains the same)
    extraction_chain = extraction_prompt | llm | parser
    
    # 3c. Invoke the chain inside the tool
    try:
        # The tool now calls the LLM for the actual work
        data = extraction_chain.invoke({"message": message})
        
        # FIX IS HERE: Access Pydantic object attributes using dot notation (.)
        return f"{data.duration} - Reason: {data.reason_category}"
        
    except Exception as e:
        # Handle cases where the internal LLM fails to produce valid JSON
        return f"Extraction tool failed: Could not reliably parse message. Error: {e}"

# 4. DEFINE THE AGENT PROMPT, TOOLS, AND EXECUTOR (Agent Setup)
# -----------------------------------------------------------------------
# ... (rest of the code is the same) ...
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert HR and AI assistant. When a user provides a slack message, use the 'getLeaveDuration' tool to extract the leave duration and reason. Only provide the extracted information."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

tools = [getLeaveDuration] 
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# 5. RUN THE AGENT WITH SAMPLE MESSAGES
# -----------------------------------------------------------------------
sample_messages = [
    "Mild fever and throat infection, out sick today.",
    "Will be OOO from 2nd Jan to 7 Jan. I need the duration.",
    "OOO for the first half - doctors appointment. What is the leave duration?",
    "Tell me a joke about cats."
]

# 6. SEND PROMPT TO LLM AND GET RESPONSE
# -----------------------------------------------------------------------
for msg in sample_messages:
    print(f"\n--- USER QUERY: {msg} ---")
    response = agent_executor.invoke({"input": msg, "chat_history": []})
    print(f"AGENT FINAL ANSWER: {response['output']}")
