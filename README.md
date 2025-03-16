# AI Agent from scratch - Research Assistant

This project implements an AI agent capable of conducting research using various tools, including web search and Wikipedia queries. The agent is built using Langchain and leverages Anthropic's Claude model for natural language understanding and generation.

## Project Structure

- `main.py`: The main script that orchestrates the research process. It takes user input, interacts with the language model and tools, and presents the structured research output.
- `requirements.txt`: Lists the project's dependencies.
- `tools.py`: Defines the tools available to the agent (web search, Wikipedia query, and file saving).
- `.env.example`: A template for the .env file, which should contain API keys for OpenAI and Anthropic. Remember to create a .env file and populate it with your actual API keys. Do not commit your .env file to version control.
- `utils.py`: Contains utility functions, currently including a function to save research output to a text file.

## Usage

The agent will attempt to answer your research query using the available tools. The output will be structured as follows:

- `topic`: The main topic of the research.
- `summary`: A summary of the findings.
- `sources`: A list of URLs or other sources used.
- `tools_used`: A list of tools utilized during the research process.

The research output is also saved to a file named `research_output.txt`.

## Setup

1. Install Dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Obtain API Keys:

- Get an API key from Anthropic (<https://anthropic.com/>).
- Create a .env file in the project's root directory and add your API key:

    ```bash
    ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"
    OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    ```

3. Run the Script:

    ```bash
    python main.py
    ```

The script will prompt you to enter a research query.
