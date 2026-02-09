# Base functions/modules
import os
from pathlib import Path
from langchain_groq import ChatGroq

from crewai import Agent, Task, Crew, Process, LLM
from typing import Type, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from crewai.tools import BaseTool
from langchain_community.vectorstores import FAISS
from tavily import TavilyClient

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

def build_crew(llm, local_tool,web_tool):
    researcher = Agent(
        role="Researcher",
        backstory="You gather evidence. You MUST use tools. You never answer from memory.",
        goal="Retrieve and compile evidence from local FAISS first, then Tavily web search only if needed.",
        llm=llm,
        tools=[local_tool, web_tool],
        verbose=False,
    )

    writer = Agent(
        role="Content Writer",
        backstory="You write clear answers grounded strictly in evidence provided.",
        goal="Write a structured answer using ONLY the Researcher's evidence. Cite evidence IDs.",
        llm=llm,
        tools=[],
        verbose=False,
    )

    critic = Agent(
        role="Reviewer",
        backstory="You are strict about grounding. No evidence = no claim.",
        goal="Check the draft for unsupported claims, missing coverage, and clarity against the evidence.",
        llm=llm,
        tools=[],
        verbose=False,
    )

    reviser = Agent(
        role="Reviser",
        backstory="You apply reviewer feedback and output the final answer.",
        goal="Revise the draft to fully comply with evidence and reviewer notes. Output final Markdown only.",
        llm=llm,
        tools=[],
        verbose=False,
    )

    t1 = Task(
        description=(
            "User query: {query}\n\n"
            "1) ALWAYS call local_search(query={query}, k=6).\n"
            "2) Decide if web search is necessary:\n"
            "   - If the query asks for recency (latest/today/current/news) OR\n"
            "   - Local evidence is insufficient/weak (few relevant chunks)\n"
            "   then call web_search(query={query}, max_results=5, search_depth='basic').\n\n"
            "Return two sections:\n"
            "A) EVIDENCE (list): each item must include an ID [L-xxxx] or [W-xxxx], plus source/url/page and a 1-sentence summary.\n"
            "B) FINDINGS (bullets): each bullet must cite one or more evidence IDs.\n"
        ),
        expected_output="Evidence list + Findings with citations like [L-0001] and [W-0002].",
        agent=researcher,
    )

    t2 = Task(
        description=(
            "Write a Markdown answer to: {query}\n"
            "Use ONLY the Researcher's EVIDENCE/FINDINGS.\n"
            "Every major claim must have citations like [L-0001] or [W-0002].\n"
            "If evidence is insufficient, say so and suggest what else is needed.\n"
        ),
        expected_output="Markdown answer with evidence-ID citations.",
        agent=writer,
        context=[t1],
    )

    t3 = Task(
        description=(
            "Critique the draft answer for: {query}\n"
            "Check that:\n"
            "- All major claims have citations.\n"
            "- No claim contradicts the evidence.\n"
            "- The answer is complete and clear.\n"
            "Output: (1) Issues list, (2) Concrete fix instructions.\n"
        ),
        expected_output="Issues + fix instructions.",
        agent=critic,
        context=[t1, t2],
    )

    t4 = Task(
        description=(
            "Revise the draft answer for: {query}\n"
            "Apply ALL fix instructions from the Reviewer.\n"
            "Keep citations. Output ONLY the final Markdown answer.\n"
        ),
        expected_output="Final Markdown answer only.",
        agent=reviser,
        context=[t1, t2, t3],
    )

    return Crew(
        agents=[researcher, writer, critic, reviser],
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: FAISS

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
    
# Internet search tools for CrewAI:
class WebSearchArgs(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(5, ge=1, le=10, description="Number of results")
    search_depth: str = Field("basic", description="basic or advanced")

class TavilyWebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web via Tavily and return relevant snippets with URLs for citation."
    args_schema: Type[BaseModel] = WebSearchArgs

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: str
    client: TavilyClient | None = None

    def model_post_init(self, __context):
        if self.client is None:
            self.client = TavilyClient(api_key=self.api_key)

    def _run(self, query: str, max_results: int = 5, search_depth: str = "basic") -> List[Dict[str, Any]]:
        res = self.client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
        )

        out: List[Dict[str, Any]] = []
        for i, r in enumerate(res.get("results", []), start=1):
            out.append({
                "id": f"W-{i:04d}",
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "text": r.get("content", "") or r.get("snippet", ""),
                "source": "tavily",
                "score": r.get("score", None),
            })
        return out