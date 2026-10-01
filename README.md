# Planet QA Agent

A question-answering system built with LangChain. An LLM decides which custom tool to call to answer questions about planets in the solar system, and a chain executes the chosen tool.

## How it works

The app is a pipeline of three chained runnables:

```
prompt template  |  LLM with tools  |  run_tools (RunnableLambda)
```

1. **Prompt template** formats the user's question.
2. **LLM with tools** (`ChatOpenAI`, `gpt-4o-mini`) picks the right tool.
3. **`run_tools`** executes the tool calls and returns the output.

### Tools

| Tool | Purpose |
| --- | --- |
| `PlanetDistanceSun` | Approximate distance from the Sun in AU (Earth, Mars, Jupiter, Pluto) |
| `PlanetRevolutionPeriod` | Approximate revolution period in Earth years (Earth, Mars, Jupiter, Pluto) |
| `PlanetGeneralInfo` | General info via similarity search (Chroma + OpenAI embeddings) over the documents in `planets/` |

Each tool takes a single `planet_name: str` argument. For unknown planets, the tools return a "not available in this tool" message.

## Project structure

```
task/
├── main.py            # tools, chain and entry point
├── planets/           # one .txt document per planet
├── test/tests.py      # stage tests (hs-test-python)
├── tests.py           # test runner
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your API key:

```
OPENAI_API_KEY=your-key-here
```

## Usage

```bash
python main.py
```

Then type a question, for example:

```
> What is the distance of Mars from the sun?
Mars is approximately 1.5 AU from the Sun.

> Tell me more about the Earth.
Earth
Earth is the third planet from the Sun and the only known planet to support life.
...
```

The program prints the tool output followed by the chain structure.

## Tests

```bash
python tests.py
```
