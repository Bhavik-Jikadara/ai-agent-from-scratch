import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables import RunnablePassthrough
import json

# Load environment variables
load_dotenv()

# Add the project root to the Python path
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Now you can import from src.tools
from src.tools import search_tool, wiki_tool, save_tool

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm your AI assistant. I can help with casual conversation or in-depth research. How can I assist you today?",
        }
    ]

if "research_mode" not in st.session_state:
    st.session_state.research_mode = False

if "llm" not in st.session_state:
    st.session_state.llm = None

# Tool initialization
tools = [search_tool, wiki_tool, save_tool]


def initialize_llm(model_provider, api_key=None, model_name=None):
    """Initialize the language model with proper error handling"""
    try:
        if model_provider == "OpenAI":
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found.")
            return ChatOpenAI(
                model=model_name or "gpt-4-0125-preview",
                api_key=api_key,
                temperature=0.7,
            )
        elif model_provider == "Anthropic":
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key not found.")
            return ChatAnthropic(
                model=model_name or "claude-3-opus-20240229",
                api_key=api_key,
                temperature=0.7,
            )
        elif model_provider == "Google":
            api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Google API key not found.")
            return ChatGoogleGenerativeAI(
                model=model_name or "gemini-pro", api_key=api_key, temperature=0.7
            )
        else:
            raise ValueError(f"Invalid model provider: {model_provider}")
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None


def detect_research_intent(query):
    """Improved research intent detection"""
    research_keywords = {
        "research",
        "study",
        "analyze",
        "investigate",
        "report",
        "data",
        "statistics",
        "survey",
        "findings",
        "sources",
        "references",
    }
    question_words = {"what", "when", "where", "why", "how", "who"}

    query_lower = query.lower()
    if any(kw in query_lower for kw in research_keywords):
        return True
    if query_lower.split()[0] in question_words and len(query.split()) > 4:
        return True
    return False


def handle_research_query(query):
    """Process research queries using tools"""
    try:
        if st.session_state.llm is None:
            st.error("Model not initialized. Check API keys in sidebar.")
            return None

        prompt = ChatPromptTemplate(
            input_variables=["query"],
            messages=[
                (
                    "system",
                    """You're a research assistant. Use tools to gather information and provide structured response with sources.""",
                ),
                ("human", "{query}"),
            ],
        )

        agent = create_tool_calling_agent(st.session_state.llm, prompt, tools)
        # The issue was here, the agent was not a runnable.
        agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=True
        ).with_config({"run_name": "AgentExecutor"})

        response = agent_executor.invoke({"query": query})
        return parse_response(response.get("output", ""))

    except Exception as e:
        st.error(f"Research error: {str(e)}")
        return None


def parse_response(response):
    """Parse and validate agent response"""
    try:
        if isinstance(response, dict):
            return response

        if isinstance(response, str):
            if response.startswith("{"):
                return json.loads(response)

            # Extract JSON from markdown if present
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
                return json.loads(response.strip())

        return {"summary": response, "sources": [], "tools_used": []}

    except json.JSONDecodeError:
        return {"summary": response, "sources": [], "tools_used": []}


def handle_conversation(query):
    """Handle general conversation"""
    try:
        if st.session_state.llm is None:
            return "Please configure the model first in the sidebar."

        messages = [
            {"role": "system", "content": "You're a helpful assistant."},
            *[
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages[-4:]
            ],
            {"role": "user", "content": query},
        ]

        return st.session_state.llm.invoke(messages).content
    except Exception as e:
        return f"Error: {str(e)}"


def display_message(msg):
    """Display a message in the chat interface."""
    if isinstance(msg["content"], dict):
        st.markdown(f"**{msg['content'].get('summary', '')}**")
        if msg["content"].get("sources"):
            with st.expander("Sources & Tools"):
                st.write("**Sources:**")
                for src in msg["content"]["sources"]:
                    st.write(f"- {src}")
                st.write("**Tools Used:**")
                for tool in msg["content"]["tools_used"]:
                    st.write(f"- {tool}")
    else:
        st.write(msg["content"])


# Streamlit UI
st.set_page_config(
    page_title="Research Assistant 🔍",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Sidebar Configuration
with st.sidebar:
    st.title("Configuration")
    model_provider = st.selectbox(
        "Model Provider", ["OpenAI", "Anthropic", "Google"], index=0
    )

    api_key = st.text_input(
        f"{model_provider} API Key",
        type="password",
        value=os.getenv(f"{model_provider.upper()}_API_KEY", ""),
    )

    model_name = st.selectbox(
        "Model Version",
        options={
            "OpenAI": [
                "gpt-4-0125-preview",
                "gpt-4-turbo-preview",
                "gpt-3.5-turbo-0125",
            ],
            "Anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
            "Google": [
                "gemini-2.0-flash",
                "gemini-pro",
                "gemini-1.5-pro-preview",
                "gemini-1.0-pro-latest",
            ],
        }[model_provider],
        index=0,
    )

    if st.button("Initialize Model"):
        try:
            st.session_state.llm = initialize_llm(model_provider, api_key, model_name)
            if st.session_state.llm is not None:
                st.success(f"{model_provider} model initialized!")
            else:
                st.error("Model initialization failed. Check API keys and model name.")
        except Exception as e:
            st.error(f"Initialization failed: {str(e)}")

    st.divider()
    st.session_state.research_mode = st.checkbox(
        "Force Research Mode",
        value=False,
        help="Always use research tools for responses",
    )


col1, col2 = st.columns(spec=[0.8, 0.15])

with col1:
    st.subheader("🔍 Research Assistant")
    st.caption("Switch between conversation and research modes using the sidebar")


# Clear chat history button
def clear_chat_history():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "What would you like me to research for you today?",
        }
    ]


with col2:
    st.button("Clear Chat", on_click=clear_chat_history)


# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        display_message(msg)

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display the user message immediately
    with st.chat_message("user"):
        st.write(prompt)

    # Determine response type
    use_research = st.session_state.research_mode or detect_research_intent(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if use_research:
                    response = handle_research_query(prompt)
                else:
                    response = handle_conversation(prompt)

                if response:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                    display_message({"content": response})
                else:
                    st.error("No response generated.")

            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
