from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing import Literal
from langchain.messages import HumanMessage


load_dotenv()

embeddings = OllamaEmbeddings(
    model="embeddinggemma",
)

# Qdrant client (local)
qdrant_client = QdrantClient(
    url="http://localhost:6333",
)

# Create / load collection
collection_name = "github_io"

vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name=collection_name,
)

retriever = vectorstore.as_retriever()

@tool
def retrieve_posts(query: str) -> str:
    "Search and return information about blog posts."
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

retriever_tool = retrieve_posts


model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# model = ChatOllama(model="qwen3:0.6b", temperature=0)

# node
def generate_query_or_respond(state: MessagesState):
    "Call the model to generate a response based on the current state. Given the question, it will decide to retrieve using the retriever tool, or simply respond to the user."
    response = (
        model.bind_tools([retriever_tool]).invoke(state["messages"])
    )
    return {"messages": [response]}

# ## Condition Edge
# In the following snippet, we push LLm to decide if the retrieved documents are relevant ot the user's question or not

system_prompt = """ You are an AI assessment assistant measuring the relevance of retrieved documents to a user question
Here are the retrieved documents \n\n {context} \n\n
Here is the user question: {question}\n
If the documents contain keyword(s) or semantic meaning related to the user's question, grade it as relevant
Given a binary score 'yes' or 'no' score to indicate whether the documents are relevant to the question."""


class RelevantDocuments(BaseModel):
    """ Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not."
    )

# node
def grade_documents(
    state: MessagesState
) -> Literal["generate_answer", "rewrite_question"]:
    "Determine whether the retrieved documents are relevant to the question."
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = system_prompt.format(question=question, context=context)

    response = model.with_structured_output(RelevantDocuments).invoke(
        [{"role": "user", "content": prompt}])

    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"


rewrite_prompt = """Look at the input and try to reason about the underlying semantic intent or meaning.\n
Here is the initial question:\n\n {question}\n\n
Rewrite an improved question"""

def rewrite_question(state: MessagesState):
    "Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = rewrite_prompt.format(question=question)
    response = model.invoke([{"role": "user", "content": prompt}])

    return {"messages": [HumanMessage(content=response.content)]}


GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node(generate_query_or_respond)
builder.add_node("retrieve", ToolNode([retriever_tool]))
builder.add_node(rewrite_question)
builder.add_node(generate_answer)

builder.add_edge(START, "generate_query_or_respond")
builder.add_conditional_edges("generate_query_or_respond", tools_condition, {"tools": "retrieve", END: END})
builder.add_conditional_edges("retrieve", grade_documents)
builder.add_edge("generate_answer", END)
builder.add_edge("rewrite_question", "generate_query_or_respond")

graph = builder.compile()




