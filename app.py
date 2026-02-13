import streamlit as st
import os

# Disable CrewAI / OTEL telemetry before importing `crewai` so it doesn't
# attempt to register signal handlers during import (Streamlit runs code
# in worker threads which can't register signals).
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")

import crewai

from pathlib import Path
from typing import Any, Dict
from crewai import Agent, Task, Crew, Process, LLM

from docstore_functions import load_docstore_from_dir
from load_keys import *
from llm_and_crew_functions import (
	LocalFAISSSearchTool,
	TavilyWebSearchTool,
	build_crew,
	load_llm,
)

DEFAULT_MODEL = "groq/openai/gpt-oss-120b"
DEFAULT_GROQ_KEY_PATH = "keys/groq.json"
DEFAULT_TAVILY_KEY_PATH = "keys/tavily.json"
DEFAULT_OPENAI_KEY_PATH = "keys/openai.json"
DEFAULT_INDEX_DIR = "vectorstore"


st.set_page_config(
	page_title="Agentic RAG App",
	page_icon="AI",
	layout="wide",
)


@st.cache_resource(show_spinner=False)
def bootstrap_resources(
	model_name: str = DEFAULT_MODEL,
	groq_key_path: str = DEFAULT_GROQ_KEY_PATH,
	openai_key_path: str = DEFAULT_OPENAI_KEY_PATH,
	tavily_key_path: str = DEFAULT_TAVILY_KEY_PATH,
	index_dir: str = DEFAULT_INDEX_DIR) -> Dict[str, Any]:
	
	"""Load the LLM, FAISS store, tools, and Crew exactly once per session."""
	
	# Load all the APIs + tools (and set their respective API keys):
	groq_api_key = load_groq_key(Path(groq_key_path))
	os.environ['GROQ_API_KEY'] = groq_api_key

	llm_for_crew = LLM(model="groq/openai/gpt-oss-120b", api_key=os.environ["GROQ_API_KEY"], temperature=0.2)

	store, docs = load_docstore_from_dir(index_dir=index_dir)
	local_tool = LocalFAISSSearchTool(store=store)

	tavily_api_key = load_tavily_key(Path(tavily_key_path))
	openai_api_key = load_openai_key(Path(openai_key_path))
	os.environ['OPENAI_API_KEY'] = openai_api_key
	web_tool = TavilyWebSearchTool(api_key=tavily_api_key)

	# 'Load' crew ready for use:
	crew = build_crew(llm=llm_for_crew, local_tool=local_tool, web_tool=web_tool)

	return {
		"crew": crew,
		"doc_count": len(docs),
		"model": model_name,
		"index_dir": index_dir,
	}


def format_crew_output(output: Any) -> str:
	"""Best-effort extraction of the final answer from Crew output objects."""

	if output is None:
		return "No answer returned."

	candidate_attrs = [
		"final_output",
		"result",
		"raw",
		"output",
		"response",
	]
	for attr in candidate_attrs:
		value = getattr(output, attr, None)
		if isinstance(value, str) and value.strip():
			return value
	if isinstance(output, str):
		return output
	return str(output)


def main() -> None:
	try:
		resources = bootstrap_resources()
	except FileNotFoundError as err:
		st.error(f"Configuration error: {err}")
		st.stop()
	except Exception as err:  # pragma: no cover - defensive catch for UI
		st.error(f"Failed to initialize the agentic stack: {err}")
		st.stop()

	st.title("Agentic RAG Crew")
	st.write(
		"Ask a question"
	)

	# Sidebar to show how many docs in FAISS are loaded
	#
	# with st.sidebar:
	#	st.subheader("System Status")
	#	st.metric("Indexed chunks", resources["doc_count"])
	#	st.caption(
	#		f"Model: {resources['model']}\nFAISS index: {resources['index_dir']}"
	#	)
	#	if st.button("Reload resources"):
	#		bootstrap_resources.clear()
	#		st.experimental_rerun()

	# Allows for the shwoing of question history:
	if "exchanges" not in st.session_state:
		st.session_state["exchanges"] = []

	query = st.text_area(
		"Your question",
		placeholder="e.g. List some common reactions in natural product synthesis",
		height=120,
	)
	run_button = st.button("Retrieve answer", type="primary", use_container_width=True)

	if run_button:
		if not query.strip():
			st.warning("Please enter a question before running.")
		else:
			with st.spinner("Retrieving answer..."):
				try:
					crew_output = resources["crew"].kickoff(inputs={"query": query.strip()})
					final_answer = format_crew_output(crew_output)
					st.session_state["exchanges"].append(
						{"question": query.strip(), "answer": final_answer}
					)
				except Exception as err:  # pragma: no cover - Crew runtime errors surface here
					st.error(f"Agent crew failed: {err}")

	for exchange in st.session_state["exchanges"]:
		with st.chat_message("user"):
			st.write(exchange["question"])
		with st.chat_message("assistant"):
			st.markdown(exchange["answer"])


if __name__ == "__main__":
	main()
