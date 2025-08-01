# 🤖 Enhanced AI Code Review Assistant

An advanced AI-powered code review system featuring comprehensive code analysis with security checks, automatic code fixes, refactoring suggestions, and insightful metrics — all wrapped in a sleek, modern web interface.

---

## 🚀 Project Overview

This project provides an AI-driven platform to analyze source code for potential issues such as security vulnerabilities, error handling problems, performance bottlenecks, and code maintainability concerns. It automatically suggests code fixes, highlights critical improvements, and refactors code where possible.

The system includes:

- **FastAPI backend** with enhanced static and dynamic code analysis.
- Intelligent **automatic code fixes** and security improvements.
- Rich **frontend UI** for submitting code and viewing detailed analysis.
- Support for multiple programming languages: Python, JavaScript, Java, C++.
- Integration with AI models (OpenAI GPT-4, local LLM) for deep analysis.
- Metrics visualization and recommendations to improve code quality.

---

## 🧩 Features

- **Security Focused:** Detects critical security vulnerabilities like use of `eval()`, `exec()`, unsafe subprocess calls, unsafe deserialization, and XSS risks.
- **Auto Fixes:** Generates suggested fixes with explanations and safe alternatives.
- **AI Summaries:** AI-powered summaries and improvement recommendations.
- **Code Refactoring:** Produces fully refactored code incorporating fixes.
- **Detailed Metrics:** Displays lines of code, comment ratios, cyclomatic complexity, and other code quality indicators.
- **Friendly UI:** Modern, responsive web frontend with live code submission and results.
- **Sample Code:** Pre-loaded sample snippets showing good and bad coding practices.

---

## 📁 Project Structure

/
├── main.py # FastAPI backend server

├── index.html # Frontend web application

├── requirements.txt # Python dependencies

├── runtime.txt # Python runtime version for deployment

└── README.md # Project documentation 


## 🛠️ Getting Started

### Prerequisites

- Python 3.11 or higher (3.11 recommended for best compatibility)
- `pip` package manager
- OpenAI API key for GPT-4-based AI analysis
- Hugging Face Transformers for local model support

### Installation

1. Clone the repository:

    ```
    git clone https://github.com/AnkitaKapoor980/ai-code-review-assistant.git
    cd ai-code-review-assistant
    ```

2. Create and activate a virtual environment:

    ```
    python3 -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```

3. Install dependencies:

    ```
    pip install -r requirements.txt
    ```

4. Set environment variables for AI API keys:

    ```
    export OPENAI_API_KEY="your_openai_api_key_here"
    export HUGGINGFACE_TOKEN="your_hf_token_here"
    ```

### Running Locally

Start the FastAPI server with auto reload:

uvicorn main:app --host 0.0.0.0 --port 8001 --reload

Visit your browser at:

http://localhost:8001

---

## 🔧 Usage

- Paste or write code in the frontend text area.
- Select the programming language.
- Toggle automatic fix generation.
- Submit for AI-powered analysis.
- Explore detailed issues, suggested fixes, refactored code, and metrics.
- Use sample buttons to load example code quickly.

---

## 🌐 Deployment

- Compatible with platforms like Render, Railway, etc.
- Use `runtime.txt` to specify Python version (3.11 recommended).
- Make sure `requirements.txt` is included.
- Deploy with commands similar to:

## LIVE WORKING LINK
    ```
    ai-code-review-assistant-production.up.railway.app
    ```

---

## 🧠 AI Models and Analysis

- Hybrid analysis combining AST parsing, pattern rules, and AI-driven suggestions.
- Optionally uses:
  - OpenAI GPT-4 for contextual understanding
  - Local Hugging Face transformers models for privacy-focused offline analysis
- Provides categorized issues with severity and secure fix suggestions.

---

## 🛡️ Security Practices

- Highlights critical security risks like `eval()`, `exec()`, unsafe subprocess calls, deserialization vulnerabilities, and XSS.
- Suggests and auto-generates fixes to improve code safety.
- Encourages robust error handling and code maintainability.

---

## 📝 Sample Code Included

- Python and JavaScript examples demonstrating security issues and best practices.
- Easy-to-load samples via frontend UI buttons for demonstration.

---

## 📋 Dependencies

- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [Pydantic](https://pydantic.dev/)
- [Transformers](https://huggingface.co/transformers/) (optional)
- [OpenAI Python SDK](https://github.com/openai/openai-python) (optional)

---


- Thanks to the FastAPI, OpenAI, and Hugging Face communities.
- UI styled using Prism.js and Google Fonts.

---

Enjoy smarter, safer, and cleaner coding with AI assistance! 🚀
