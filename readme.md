## Setting up Virtual Environment

# For Linux Users:

    First create a virtual environment using:
          python3 -m venv .venv

    Then activate it using:
        source .venv/bin/activate

# For Windows Users:

    First create a virtual environment using:
          python -m venv .venv

    Then activate it using:
        source .venv/Scripts/activate

## Installing Olama LLM

# For Linux Users:

    curl -fsSL https://ollama.ai/install.sh | sh

# For Windows Users

    Download from "https://ollama.ai/download/windows"

# pull ollama model

    ollama pull mistral

# test if ollama is working

    ollama run mistral "Hello, are you working?"
