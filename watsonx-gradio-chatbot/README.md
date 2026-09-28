# IBM watsonx.ai Generative AI Chatbot

A lightweight **LLM-powered web chatbot** built with **IBM watsonx.ai**, **LangChain**, and **Gradio**.

This project demonstrates how a foundation model hosted on IBM watsonx.ai can be wrapped with LangChain and exposed through an interactive browser-based interface.

> **Project origin:** This project was developed as part of hands-on Coursera coursework and was independently organized, documented, and extended for portfolio and GitHub presentation.

## Project Overview

The application accepts a natural-language prompt from a user, sends it to a foundation model hosted on IBM watsonx.ai, and displays the generated response in a Gradio web interface.

The default configuration uses:

- **IBM watsonx.ai** for foundation-model inference
- **Meta Llama 4 Maverick** as the language model
- **LangChain IBM** as the LLM integration layer
- **Gradio** for the interactive web application
- **Python** for application logic

## Architecture

```text
User
  |
  v
Gradio Web Interface
  |
  v
generate_response()
  |
  v
LangChain WatsonxLLM
  |
  v
IBM watsonx.ai Foundation Model
  |
  v
Generated Response
  |
  v
Gradio Output
```

## Features

- Interactive browser-based generative AI interface
- IBM watsonx.ai foundation-model integration
- LangChain-based LLM abstraction
- Configurable generation parameters
- Environment-variable support for project configuration
- Input validation and basic exception handling
- Example prompts for quick testing

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| IBM watsonx.ai | Foundation-model inference |
| Meta Llama 4 Maverick | Large language model |
| LangChain IBM | LLM integration |
| Gradio | Web-based user interface |

## Repository Structure

```text
watsonx-gradio-chatbot/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/watsonx-gradio-chatbot.git
cd watsonx-gradio-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the watsonx.ai project

Copy `.env.example` to `.env` or set the variables directly in your environment.

```env
WATSONX_PROJECT_ID=your_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_MODEL_ID=meta-llama/llama-4-maverick-17b-128e-instruct-fp8
```

Authentication requirements can vary depending on the IBM watsonx.ai environment you are using. Configure your IBM Cloud/watsonx credentials according to your environment before running the application.

## Run the Application

```bash
python app.py
```

Open the local Gradio URL shown in the terminal. By default, the application runs at:

```text
http://127.0.0.1:7860
```

## How It Works

1. A user enters a question in the Gradio interface.
2. `generate_response()` receives the prompt.
3. LangChain's `WatsonxLLM` sends the prompt to IBM watsonx.ai.
4. The selected foundation model generates a response.
5. The response is returned to the Gradio interface.

## Generation Configuration

The application currently uses:

```python
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.5
```

A lower temperature generally produces more predictable responses, while a higher temperature can increase response variation.

## Example Use Cases

- General question answering
- Generative AI demonstrations
- Education and concept explanation
- Rapid LLM interface prototyping
- Foundation-model experimentation
- Testing prompt-response behavior

## What I Learned

Through this project, I practiced:

- Connecting applications to hosted foundation models
- Using LangChain abstractions for LLM inference
- Designing a lightweight AI user interface with Gradio
- Configuring LLM generation parameters
- Structuring a small generative-AI application for reproducibility
- Separating application configuration from source code

## Possible Future Improvements

- Convert the single-turn interface into a multi-turn chatbot
- Add conversation memory
- Add streaming responses
- Support multiple watsonx.ai models
- Add prompt templates and system instructions
- Add Retrieval-Augmented Generation (RAG)
- Add document upload and question answering
- Add response evaluation and logging
- Containerize the application with Docker
- Deploy the interface to a cloud platform

## Portfolio Summary

**IBM watsonx.ai Generative AI Chatbot** — Built an interactive LLM application using Python, IBM watsonx.ai, LangChain, and Gradio. Integrated a Meta Llama foundation model through LangChain's WatsonxLLM interface, configured generation parameters, and developed a browser-based prompt/response workflow for real-time generative AI interaction.

## Skills Demonstrated

`Python` · `Generative AI` · `LLMs` · `IBM watsonx.ai` · `LangChain` · `Gradio` · `Prompt Engineering` · `AI Application Development`

## Acknowledgment

The initial learning exercise was completed through Coursera course material. The repository structure, documentation, code organization, configuration improvements, and portfolio presentation were prepared as part of independent learning and project development.
