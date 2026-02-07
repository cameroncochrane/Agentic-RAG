# Base functions/modules
import os
from pathlib import Path
from langchain_groq import ChatGroq

from crewai import Agent, Task, Crew, Process, LLM
from typing import Type, List, Dict, Any
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from langchain_community.vectorstores import FAISS

# Bespoke functions/modules:
from load_keys import *
from docstore_functions import *


def load_llm(model_type:str = 'openai/gpt-oss-120b',api_key_path:str = "keys/groq.json"):
    
    p = Path(api_key_path)

    groq_api_key = load_groq_key(p)
    os.environ['GROQ_API_KEY'] = groq_api_key
    
    model = ChatGroq(
            model=model_type,
            temperature=0.2,
            max_retries=2,
        )
    
    return model

def give_message(model,message):

    full_answer = model.invoke(message)
    extracted_answer = full_answer.content

    return extracted_answer

def ask_FAISS_with_LLM(llm,store, question: str, k: int = 6):
    docs = store.similarity_search(question, k=k)

    context = "\n\n".join(
        [f"[{i+1}] Source={d.metadata.get('source','')} Page={d.metadata.get('page', '')}\n{d.page_content}"
         for i, d in enumerate(docs)]
    )

    prompt = f"""Use ONLY the context to answer the question.
    If the answer is not in the context, say "I don't know based on the provided documents."

    Context:
    {context}

    Question: {question}
    Answer:"""

    resp = llm.invoke(prompt)
    return resp.content, docs

# Crew functions:
def build_test_crew(llm):
    researcher = Agent(
        role="Researcher",
        backstory="",
        goal="Retrieve and compile evidence using the provided LLM as the sole knowledge source (do NOT call external/local tools).",
        llm=llm,
        tools=[],
        verbose=False,
    )

    writer = Agent(
        role="Content Writer",
        backstory="",
        goal="Write a structured answer grounded in the evidence produced by the Researcher (LLM).",
        llm=llm,
        tools=[],
        verbose=False,
    )

    critic = Agent(
        role="Reviewer",
        backstory="",
        goal="Check the answer for unsupported claims, missing info, and clarity against the Researcher's evidence.",
        llm=llm,
        tools=[],
        verbose=False,
    )

    t1 = Task(
        description=(
            "Research this query using only the LLM: {query}\nReturn bullet findings." # It's important to add the specific variable containing the query hereQ!!
        ),
        expected_output="A bullet list of findings derived from the LLM only.",
        agent=researcher,
    )

    t2 = Task(
        description=(
            "Write the answer. Use the Researcher's findings as the source."
        ),
        expected_output="A Markdown response mapped to the Researcher's findings.",
        agent=writer,
        context=[t1],
    )

    t3 = Task(
        description=(
            "Critique the answer against the Researcher's response. List issues and provide fix instructions."
        ),
        expected_output="A list of issues found plus concrete fix instructions.",
        agent=critic,
        context=[t1, t2],
    )

    t4 = Task(description=(
            "Consider the issues and fix instructions provided by the critic and write a final answer to be outputted to the user"
        ),
        expected_output="A Final answer, written in markdown, which considers the points raised by the critic",
        agent=writer,
        context=[t1, t2, t3],)

    return Crew(
        agents=[researcher, writer, critic],
        tasks=[t1, t2, t3, t4],
        process=Process.sequential,
        verbose=False,
    )

# Local search tools for CrewAI:
class LocalSearchArgs(BaseModel):
    query: str = Field(..., description="User question")
    k: int = Field(6, ge=1, le=20, description="Top-k chunks")

class LocalFAISSSearchTool(BaseTool):
    name: str = "local_search"
    description: str = "Search the local FAISS vectorstore and return relevant chunks with source metadata."
    args_schema: Type[BaseModel] = LocalSearchArgs

    def __init__(self, store: FAISS):
        super().__init__()
        self.store = store

    def _run(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        docs = self.store.similarity_search(query, k=k)
        out = []
        for i, d in enumerate(docs, start=1):
            out.append({
                "id": f"L-{i:04d}",
                "text": d.page_content,
                "source": d.metadata.get("source", ""),
                "page": d.metadata.get("page", None),
                "content_hash": d.metadata.get("content_hash", None),
            })

        return out