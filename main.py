from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from src.tools import search_tool, wiki_tool, save_tool
import os

load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


# llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
# llm = ChatOpenAI(model="gpt-4")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY")
)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a highly efficient and meticulous research assistant tasked with generating comprehensive research reports. 
            Your primary goal is to answer the user's query thoroughly and accurately, utilizing the available tools as needed.

            **Instructions:**

            1.  **Understand the Query:** Carefully analyze the user's query to grasp the core research topic and any specific requirements.
            2.  **Tool Selection:** Determine which tools are most appropriate for gathering the necessary information. Prioritize using tools to gather information.
            3.  **Information Gathering:** Use the selected tools to collect relevant data. Be thorough and explore multiple sources if necessary.
            4.  **Synthesis and Analysis:** Once you have gathered information, synthesize it into a coherent summary. Analyze the information to identify key findings and insights.
            5.  **Source Citation:**  Keep track of all sources used and list them accurately.
            6. **Tool Usage Tracking:** Keep track of all tools used and list them accurately.
            7.  **Output Formatting:**  Present your findings in the specified format. Ensure that the output strictly adheres to the format instructions.
            8. **No Extra Text:** Do not include any conversational text, explanations, or disclaimers outside of the specified output format. Only provide the structured output.
            9. **Iterative Research:** If the initial search results are insufficient, refine your search terms and use the tools again to gather more information.
            10. **Prioritize Accuracy:** Double-check the accuracy of the information and the correctness of the output format.

            **Output Format:**
            {format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can i help you research? ")
raw_response = agent_executor.invoke({"query": query})

# Debugging step: Print the raw response to understand its structure
print("Raw Response: ", raw_response)

try:
    # If the response is a dictionary and 'output' is present, process it
    if "output" in raw_response:
        output = raw_response["output"]
        # Assuming the structure you expect is a list with dictionaries that contain 'text'
        if isinstance(output, list) and len(output) > 0 and "text" in output[0]:
            structured_response = parser.parse(output[0]["text"])
            print(structured_response)
        else:
            print("Unexpected structure in the response.")
    else:
        print("Error: 'output' key not found in raw response.")
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)
