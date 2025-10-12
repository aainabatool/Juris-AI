# agentic_rag.py
# Clean version – wrapped into a single callable function `run_agentic_rag(query)`

import os
import re
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain import hub
from langchain.schema import HumanMessage

# ==========================================================
# 1. Load environment and setup
# ==========================================================
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# ==========================================================
# 2. Setup vector store retriever
# ==========================================================
persist_directory = "Data/legal_ai_db"
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store_langgraph = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)
retriever_langgraph = vector_store_langgraph.as_retriever(search_kwargs={"k": 5})
langgraph_retriever_tool = create_retriever_tool(
    retriever=retriever_langgraph,
    name="retriever_vectordb_langgraph",
    description="Useful for retrieving information from the Pakistani legal database."
)

# ==========================================================
# 3. Define the agent + helper functions
# ==========================================================

def agent(state: MessagesState):
    """LLM decision node — decides whether to call tools or respond directly."""
    print("---CALL AGENT---")
    messages = state['messages']
    model = ChatGroq(model_name="openai/gpt-oss-20b").bind_tools([langgraph_retriever_tool])
    response = model.invoke(messages)
    return {'messages': [response]}


def grade_documents(state: MessagesState) -> Literal['generate', 'rewrite']:
    """Grades retrieved docs to decide whether to generate or rewrite."""
    print("---CALL GRADE DOCUMENTS---")

    class Grade(BaseModel):
        binary_score: str = Field(description="Score for relevance 'yes' or 'no'")

    llm = ChatGroq(model_name="openai/gpt-oss-20b").with_structured_output(Grade)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a grader assessing relevance of a retrieved document to a user question.\n\n"
            "Document:\n{context}\n\n"
            "Question: {question}\n"
            "Respond with 'yes' if relevant, otherwise 'no'."
        )
    )

    chain = prompt | llm
    messages = state['messages']
    question = messages[0].content
    context = messages[-1].content

    scored_result = chain.invoke({"context": context, "question": question})
    score = scored_result.binary_score.strip().lower()

    if score == 'yes':
        print("✅ Document is relevant — generating content.")
        return 'generate'
    else:
        print("🔁 Document not relevant — rewriting query.")
        return 'rewrite'


def generate(state: MessagesState) -> dict:
    """Generate answer from relevant docs."""
    print("---CALL GENERATE---")
    messages = state['messages']
    question = messages[0].content
    docs = messages[-1].content

    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatGroq(model_name="openai/gpt-oss-20b")
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs, "question": question})
    return {'messages': [response]}


def rewrite(state: MessagesState) -> dict:
    """Rewrites query if retrieved context not relevant."""
    print("---CALL REWRITE---")
    question = state['messages'][0].content

    msg = [
        HumanMessage(content=(
            f"Analyze and improve the semantic clarity of the following question:\n\n{question}\n\n"
            "Return a clearer, more specific rephrasing."
        ))
    ]

    model = ChatGroq(model_name="openai/gpt-oss-20b")
    response = model.invoke(msg)
    return {'messages': [response]}


# ==========================================================
# 4. Define the graph workflow
# ==========================================================
workflow = StateGraph(MessagesState)
workflow.add_node('LLM Decision Maker', agent)
workflow.add_node('Vector Retriever', ToolNode([langgraph_retriever_tool]))
workflow.add_node('Query Rewriter', rewrite)
workflow.add_node('Content Generator', generate)

workflow.add_edge(START, 'LLM Decision Maker')
workflow.add_conditional_edges(
    'LLM Decision Maker',
    tools_condition,
    {"tools": "Vector Retriever", END: END}
)
workflow.add_conditional_edges(
    'Vector Retriever',
    grade_documents,
    {'generate': 'Content Generator', 'rewrite': 'Query Rewriter'}
)
workflow.add_edge('Content Generator', END)
workflow.add_edge('Query Rewriter', 'LLM Decision Maker')

graph = workflow.compile()

# ==========================================================
# 5. Main callable function
# ==========================================================
def run_agentic_rag(query: str):
    """
    Executes the Agentic RAG pipeline for a given query.
    Returns a JSON response with the query and generated answer.
    """
    print(f"\n🧠 Running Agentic RAG for query: {query}\n")

    think_pattern = re.compile(r"<think>.*?</think>", re.DOTALL)
    initial_input = {"messages": [HumanMessage(content=query)]}
    final_response = ""

    for event in graph.stream(initial_input, stream_mode="values"):
        if 'messages' in event:
            last_msg = event['messages'][-1].content
            clean_content = re.sub(think_pattern, '', last_msg).strip()
            final_response = clean_content
            print("================================ Cleaned Message =================================")
            print(clean_content)

    # Return JSON-compatible dict
    return {
        "query": query,
        "response": final_response
    }
