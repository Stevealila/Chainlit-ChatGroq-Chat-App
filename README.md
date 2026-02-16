# Code-y AI

Get started with `chainlit` using the `ChatGroq` opensource LLM.

1. Clone the repo.

```
git clone https://github.com/stevealila/Chainlit-ChatGroq-Chat-App.git
```

2. Grab your [Groq API key](https://console.groq.com/keys) and store it in a `.env` file (see `.env.example`).

3. Install dependencies and activate the virtual environment.

```
uv sync
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

4. Run

```
chainlit run main.py
```

And voila, you are chatting with code-y AI!
