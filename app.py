from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os

# 1. SETUP - Load Environment and Initialize Model
# -----------------------------------------------------------------------
load_dotenv()
#print(os.environ.get("OPENAI_API_KEY"))
llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")
#llm = ChatOpenAI(temperature=0.9, model="gpt-5-mini")
#llm = ChatOllama(temperature=0.0, model="llama-4-8b-instant")

print("Hello ai session!!!")

# 2. DEFINE VARIABLES
# -----------------------------------------------------------------------
var_info = """
cat
"""

# 3. CREATE PROMPT
# -----------------------------------------------------------------------
system_prompt = """
Tell me joke about {var_info}
"""

final_prompt_template = PromptTemplate(
    input_variable=["var_info"],
    template=system_prompt
)

# 4. SEND PROMPT TO LLM
# -----------------------------------------------------------------------
chain = final_prompt_template | llm
response = chain.invoke(input={"var_info": var_info})

# 5. PRINT RESPONSE
# -----------------------------------------------------------------------
# To see the output:
print(response.content)

