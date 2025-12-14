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
summary_template = """
Tell me joke about {var_info}
"""

summary_prompt_template = PromptTemplate(
    input_variable=["var_info"], template=summary_template
)

#prompt_template = PromptTemplate.from_template("Tell me joke about {topic}")
#prompt_template.invoke({"topic": "cats"})

# 4. SEND PROMPT TO LLM
# -----------------------------------------------------------------------
chain = summary_prompt_template | llm
response = chain.invoke(input={"var_info": var_info})

# 5. PRINT RESPONSE
# -----------------------------------------------------------------------
# To see the output:
print(response.content)

#def main():
#    print("Hello iaisession!!!")

#if __name__ == "__main__":
#    main()
