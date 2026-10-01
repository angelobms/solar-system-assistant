# install the required packages:
# pip install langchain-core python-dotenv langchain-openai langchain-chroma chromadb

import dotenv

dotenv.load_dotenv()
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

PLANETS_DIR = Path(__file__).parent / "planets"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)  # if using a different base_url and model, pass them as well

documents = [
    Document(page_content=path.read_text(encoding="utf-8").strip(), metadata={"source": path.name})
    for path in sorted(PLANETS_DIR.glob("*.txt"))
]

vector_store = Chroma.from_documents(
    documents,
    embeddings,
    collection_name="planets",
)


DISTANCES = {
    "earth": "Earth is approximately 1 AU from the Sun.",
    "mars": "Mars is approximately 1.5 AU from the Sun.",
    "jupiter": "Jupiter is approximately 5.2 AU from the Sun.",
    "pluto": "Pluto is approximately 39.5 AU from the Sun.",
}

PERIODS = {
    "earth": "Earth takes approximately 1 Earth year to revolve around the Sun.",
    "mars": "Mars takes approximately 1.88 Earth years to revolve around the Sun.",
    "jupiter": "Jupiter takes approximately 11.86 Earth years to revolve around the Sun.",
    "pluto": "Pluto takes approximately 248 Earth years to revolve around the Sun.",
}


@tool("PlanetDistanceSun")
def planet_distance_sun(planet_name: str) -> str:
    """Get the approximate distance of a planet from the Sun in Astronomical Units (AU)."""
    return DISTANCES.get(
        planet_name.strip().lower(),
        f"Information about the distance of {planet_name} from the Sun is not available in this tool.",
    )


@tool("PlanetRevolutionPeriod")
def planet_revolution_period(planet_name: str) -> str:
    """Get the approximate revolution period of a planet around the Sun in Earth years."""
    return PERIODS.get(
        planet_name.strip().lower(),
        f"Information about the revolution period of {planet_name} is not available in this tool.",
    )


@tool("PlanetGeneralInfo")
def planet_general_info(planet_name: str) -> str:
    """Get general information about a planet (anything except its distance from the Sun
    or its revolution period) by searching the planet documents."""
    name = planet_name.strip().lower()
    for doc in vector_store.similarity_search(f"interesting facts about {planet_name}", k=3):
        if Path(doc.metadata["source"]).stem == name:
            return doc.page_content
    return f"Additional information for {planet_name} is not available in this tool."


tools = [planet_distance_sun, planet_revolution_period, planet_general_info]
tools_by_name = {t.name: t for t in tools}

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")).bind_tools(tools)

def run_tools(message) -> str:
    return "\n".join(
        tools_by_name[call["name"]].invoke(call["args"]) for call in message.tool_calls
    )

prompt = PromptTemplate.from_template(
    "You are a helpful assistant who answers questions users may have. You are asked: {question}."
)
chain = prompt | llm | RunnableLambda(run_tools)

query = input().strip()
print(chain.invoke({"question": query}))
print()
print(chain)
