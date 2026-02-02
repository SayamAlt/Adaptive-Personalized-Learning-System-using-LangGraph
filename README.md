# 🧠 Adaptive Personalized Learning System

An AI-powered, interactive learning platform that adapts to your learning style, background, and goals. Built with **Streamlit**, **LangGraph**, and **OpenAI**, this system simulates a personalized tutor that creates tailored lesson plans, assesses your understanding, and dynamically adjusts the curriculum.

## ✨ Key Features

*   **Personalized Curriculum**: Generates lessons based on your *Learning Style* (Active, Reflective, etc.), *Background Level*, and *Interests*.
*   **Dynamic Adaptation**: Uses **LangGraph** to create a stateful learning loop. It assesses your performance after every lesson and decides whether to:
    *   **Reinforce**: Provide more examples.
    *   **Reteach**: Explain concepts from first principles if you're stuck.
    *   **Advance**: Move to the next topic with a "Graduation Report".
*   **Multi-Source Citations**: Automatically aggregates and cites resources from:
    *   **Arxiv** (Direct API integration for research papers)
    *   **Google Scholar** (Academic Articles)
    *   **YouTube** (Video Tutorials)
    *   **Web Search** (Tavily)
*   **Interactive Exercises**: Generates coding exercises (e.g., PyTorch/TensorFlow) relevant to the topic.
*   **Persistent Memory**: Remembers your progress, scores, and knowledge gaps across the session.

## 🏗️ Architecture

The system is split into two main components:

1.  **Frontend (`app.py`)**: A **Streamlit** dashboard that:
    *   Captures user profile data.
    *   Renders generated lessons, code snippets, and specific citations.
    *   Visualizes assessment scores (Confidence, Accuracy).
    *   Manages session state and user interactions.

2.  **Backend Logic (`workflow.py`)**: A **LangGraph** workflow that manages the learning state.
    *   **Nodes**: `generate_lesson`, `assess_learning`, `evaluate_next_step`.
    *   **Tools**: Integrates `TavilySearch`, `GoogleScholar`, `YouTubeSearch`, and `arxiv` library.
    *   **LLM**: Uses `gpt-4o-mini` for high-quality, pedagogical content generation.

## 🚀 Setup & Installation

### Prerequisites
*   Python 3.10+
*   OpenAI API Key
*   Tavily API Key (for web search)
*   SerpAPI Key (for Google Scholar)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/topic-learning-agent.git
    cd topic-learning-agent
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables**:
    Create a `.env` file or set up `.streamlit/secrets.toml` with the following keys:
    ```toml
    OPENAI_API_KEY="sk-..."
    TAVILY_API_KEY="tvly-..."
    SERP_API_KEY="..."
    ```

### Running the App

Run the Streamlit application:
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

## 🛠️ Usage

1.  **Create Profile**: On the sidebar, select your Learning Style, Experience Level, and Goal.
2.  **Choose Topic**: Enter a topic you want to master (e.g., "Attention is all you need", "Quantum Computing").
3.  **Start Journey**: Click "Generate Teaching Plan".
4.  **Learn & Interact**: Read the lesson, review citations, and solve compliance exercises.
5.  **Assess**: Click "Assess Understanding" to take a quiz. The AI will grade you and determine your next step.

