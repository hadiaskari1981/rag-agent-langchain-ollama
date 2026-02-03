
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode
from langchain.tools import tool
from typing import Literal
from dotenv import load_dotenv
from enum import Enum
from datetime import datetime

load_dotenv()

# Embeddings
embeddings = OllamaEmbeddings(model="embeddinggemma")

collection_name = "internal_docs"
vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name=collection_name,
    content_payload_key="text"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# model = ChatOllama(model="llama3.1:8b", temperature=0)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

from typing import Annotated
class RouteNames(str, Enum):
    data = "data"
    docs = "docs"
    both = "both"
    reasoning = "reasoning"

class DecisionSchema(BaseModel):
    route: RouteNames = Field(description="Routing decision.")

def pick_latest(x, y):
    return y

class CopilotState(MessagesState):
    question: Annotated[str, pick_latest] = ""  # Original user question
    route: Annotated[str, pick_latest] = "" # Routing decision
    answer: Annotated[str, pick_latest] = ""  # Final answer
    rewrite_count: int = 0  # Guard against infinite RAG rewrites
def initialize_state(state: CopilotState):
    """Initialize question from messages if not provided."""
    if not state.get("question") and state.get("messages"):
        # Extract question from first HumanMessage
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                return {"question": msg.content}
    return {}

# Update main graph

decision_prompt = """You are a router.

Your job is to choose how to answer the question.
You are NOT answering the question.

Choose ONE option:
- data       → if numbers, metrics, counts, or trends are needed
- docs       → if company documents, policies, or explanations are needed
- both       → if BOTH data metrics AND contextual documents are needed to answer completely
- reasoning  → if no external information is needed

Question:
{question}

------------------------------
Examples:

Example 1: "What is the company's refund policy?"
Why: Policy → document, No numbers required
Correct route: docs

Example 2: "How many active users did we have last week?"
Why: Explicit count, time window, no policy or explanation
Correct route: data

Example 3: "Rewrite this sentence to sound more professional."
Why: Language task, no external info
Correct route: reasoning

Example 4: "Generally what are the common reasons for customer churn?"
Why: Company knowledge/handbook/policies/analysis docs
Correct route: docs

Example 5: "Why did customer churn increase in December after the pricing change?"
Why: Needs metrics (churn numbers in December) AND context (pricing change details)
Correct route: both

Example 6: "What caused the drop in weekly active users after the deployment?"
Why: Needs metrics (user numbers) AND incident/release docs (deployment context)
Correct route: both

----------------------------------

If the question asks for company-specific explanations or guidance, prefer docs over reasoning.

Answer with ONLY ONE WORD:
data
docs
both
reasoning
"""

def route_decision_making(state: CopilotState):
    question = state["question"]
    system_prompt = decision_prompt.format(question=question)
    structured_llm = model.with_structured_output(DecisionSchema)
    response = structured_llm.invoke([SystemMessage(content=system_prompt)])
    return {"route": response.route.value}  # Store as string

def router(state: CopilotState) -> Literal["sql_agent", "rag_agent", "both_agent", "reasoning"]:
    route = state["route"]
    if route == "data":
        return "sql_agent"
    elif route == "docs":
        return "rag_agent"
    elif route == "both":
        return "both_agent"
    else:
        return "reasoning"

reasoning_prompt = """You are a helpful, knowledgeable AI assistant. You read the user demand and provide a clear response. 
User input: {question}"""

def respond_directly(state: CopilotState):
    """Answer questions using the LLM's knowledge and reasoning."""
    question = state["question"]
    system_prompt = reasoning_prompt.format(question=question)

    # Use existing messages or create new
    messages = state.get("messages", [HumanMessage(content=question)])
    response = model.invoke([SystemMessage(content=system_prompt)] + messages)

    return {
        "answer": response.content,
        "messages": [response]
    }


@tool
def retrieve_internal_docs(query: str) -> str:
    """Search and return the docs based on the provided query."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

def retrieve_docs(state: CopilotState):
    """Always retrieve docs for the given question."""
    question = state["question"]
    tool_call = {
        "name": "retrieve_internal_docs",
        "args": {"query": question},
        "id": "retrieve_docs_call",
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])
    tool_message = retrieve_internal_docs.invoke({"query": question})
    return {"messages": state.get("messages", []) + [tool_call_message, tool_message]}

class RelevantDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not."
    )

def grade_documents(state: CopilotState) -> Literal["generate_answer", "rewrite_question", "fallback_no_docs"]:
    """Determine whether the retrieved documents are relevant to the question."""
    messages = state["messages"]
    question = state["question"]

    # Get the last tool result
    context = messages[-1].content if messages else ""

    system_prompt = f"""You are an AI assessment assistant measuring the relevance of retrieved documents to a user question.
Here are the retrieved documents:
{context}

Here is the user question: {question}

If the documents contain keyword(s) or semantic meaning related to the user's question, grade it as relevant.
Given a binary score 'yes' or 'no' score to indicate whether the documents are relevant to the question."""

    response = model.with_structured_output(RelevantDocuments).invoke(
        [{"role": "user", "content": system_prompt}])

    if response.binary_score == "yes":
        return "generate_answer"

    if state.get("rewrite_count", 0) >= 2:
        return "fallback_no_docs"

    return "rewrite_question"

def rewrite_question(state: CopilotState):
    """Rewrite the original user question."""
    question = state["question"]
    prompt = f"""Look at the input and try to reason about the underlying semantic intent or meaning.
Here is the initial question:
{question}

Rewrite an improved question."""

    response = model.invoke([{"role": "user", "content": prompt}])
    return {
        "question": response.content,
        "messages": [HumanMessage(content=response.content)],
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }

def generate_answer(state: CopilotState):
    """Generate an answer from retrieved documents."""
    question = state["question"]
    messages = state["messages"]
    context = messages[-1].content if messages else ""

    prompt = f"""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}"""

    response = model.invoke([{"role": "user", "content": prompt}])
    return {"answer": response.content, "messages": [response]}

def fallback_no_docs(state: CopilotState):
    """Fallback when no relevant docs are found after rewrites."""
    msg = (
        "I couldn't find relevant internal documents to answer this. "
        "If you can share more specifics (document name, team, or exact policy area), I can try again."
    )
    response = AIMessage(content=msg)
    return {"answer": response.content, "messages": [response]}


qdrant_builder = StateGraph(CopilotState)

qdrant_builder.add_node("retrieve_docs", retrieve_docs)
qdrant_builder.add_node("retrieve", ToolNode([retrieve_internal_docs]))
qdrant_builder.add_node("rewrite_question", rewrite_question)
qdrant_builder.add_node("generate_answer", generate_answer)
qdrant_builder.add_node("fallback_no_docs", fallback_no_docs)

qdrant_builder.add_edge(START, "retrieve_docs")
qdrant_builder.add_edge("retrieve_docs", "retrieve")
qdrant_builder.add_conditional_edges(
    "retrieve",
    grade_documents,
    {"generate_answer": "generate_answer", "rewrite_question": "rewrite_question", "fallback_no_docs": "fallback_no_docs"},
)
qdrant_builder.add_edge("generate_answer", END)
qdrant_builder.add_edge("fallback_no_docs", END)
qdrant_builder.add_edge("rewrite_question", "retrieve_docs")

# qdrant_compiled = qdrant_builder.compile()

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

url = "postgresql+psycopg2://postgres:password@localhost:5432/postgres"
db = SQLDatabase.from_uri(url)
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

today = datetime.today()

get_schema_tool = next(t for t in tools if t.name == "sql_db_schema")
get_schema_node = ToolNode([get_schema_tool], name="get_schema")

run_query_tool = next(t for t in tools if t.name == "sql_db_query")
run_query_node = ToolNode([run_query_tool], name="run_query")

def list_tables(state: CopilotState):
    """List available database tables."""
    messages = state.get("messages", [])

    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "list_tables_call",
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])

    list_tables_tool = next(t for t in tools if t.name == "sql_db_list_tables")
    tool_message = list_tables_tool.invoke(tool_call)

    return {"messages": messages + [tool_call_message, tool_message]}

def call_get_schema(state: CopilotState):
    """Get schema for relevant tables."""
    llm_with_tools = model.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

generate_sql_query_system_prompt = f"""
You are an agent designed to interact with an SQL database.
Given an input question, create a syntactically correct {db.dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most 5 results.

Today is {today}. This will help you to create the filter if a specific time or time period is mentioned in the question.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table;
Only ask for the relevant columns given the question.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP, etc.) to the database.
"""

def generate_sql_query_or_respond(state: CopilotState):
    if not state.get("messages"):
        messages = [HumanMessage(content=state["question"])]
    else:
        messages = state["messages"]

    system_message = {"role": "system", "content": generate_sql_query_system_prompt}
    llm_with_tools = model.bind_tools([run_query_tool])
    response = llm_with_tools.invoke([system_message] + messages)

    if not response.tool_calls:
        return {"messages": [response], "answer": response.content}
    return {"messages": [response]}

check_query_system_prompt = f"""
You are a SQL expert with a strong attention to detail.
Double-check the {db.dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If any of the above mistakes are present, rewrite the query. If there are no mistakes, reproduce the original query.
You will call the appropriate tool to execute the query after running this check.
"""

def check_query(state: CopilotState):
    system_message = {
        "role": "system",
        "content": check_query_system_prompt,
    }

    # Generate an artificial user message to check
    tool_call = state["messages"][-1].tool_calls[0]
    user_message = {"role": "user", "content": tool_call["args"]["query"]}
    llm_with_tools = model.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    response.id = state["messages"][-1].id

    return {"messages": [response]}

def should_continue(state: CopilotState) -> Literal[END, "check_query"]:
    """Determine if we should continue with query execution."""
    messages = state["messages"]
    last_message = messages[-1]
    return "check_query" if last_message.tool_calls else END

# Build SQL agent graph
sql_builder = StateGraph(CopilotState)

sql_builder.add_node("list_tables", list_tables)
sql_builder.add_node("call_get_schema", call_get_schema)
sql_builder.add_node("get_schema", get_schema_node)
sql_builder.add_node("generate_sql_query_or_respond", generate_sql_query_or_respond)
sql_builder.add_node("check_query", check_query)
sql_builder.add_node("run_query", run_query_node)

sql_builder.add_edge(START, "list_tables")
sql_builder.add_edge("list_tables", "call_get_schema")
sql_builder.add_edge("call_get_schema", "get_schema")
sql_builder.add_edge("get_schema", "generate_sql_query_or_respond")
sql_builder.add_conditional_edges("generate_sql_query_or_respond", should_continue)
sql_builder.add_edge("check_query", "run_query")
sql_builder.add_edge("run_query", "generate_sql_query_or_respond")

# sql_compiled = sql_builder.compile()

# # Hybrid agent
# For questions requiring both SQL query and vector search

def both_agent(state: CopilotState):
    """Fan-out node for running both SQL and RAG agents."""
    return {}

def check_if_synthesis_needed(state: CopilotState):
    """Only synthesize when both agents have run."""
    route = state["route"]
    return "synthesize" if route == "both" else "end"

def synthesize_results(state: CopilotState):
    """Reduce: Combine results from multiple agents."""
    question = state["question"]
    messages = state["messages"]

    # Extract contexts from both agents
    contexts = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content:
            contexts.append(msg.content)

    combined_context = " ".join(contexts)

    synthesis_prompt = f"""Answer the user's question by synthesizing information from multiple sources:

User Question: {question}

Available Information:
{combined_context}

Provide a comprehensive answer that integrates all relevant information.
Keep your answer concise but complete (3-5 sentences)."""

    response = model.invoke([{"role": "user", "content": synthesis_prompt}])
    return {"answer": response.content, "messages": [response]}


main_builder = StateGraph(CopilotState)

main_builder.add_node("initialize", initialize_state)
main_builder.add_node("route_decision_making", route_decision_making)
main_builder.add_node("sql_agent", sql_builder.compile())
main_builder.add_node("rag_agent", qdrant_builder.compile())
main_builder.add_node("both_agent", both_agent)
main_builder.add_node("reasoning", respond_directly)
main_builder.add_node("synthesize", synthesize_results)

main_builder.add_edge(START, "initialize")
main_builder.add_edge("initialize", "route_decision_making")
main_builder.add_conditional_edges(
    "route_decision_making",
    router,
    {
        "sql_agent": "sql_agent",
        "rag_agent": "rag_agent",
        "both_agent": "both_agent",
        "reasoning": "reasoning",
    },
)
main_builder.add_edge("both_agent", "sql_agent")
main_builder.add_edge("both_agent", "rag_agent")
main_builder.add_conditional_edges(
    "sql_agent",
    check_if_synthesis_needed,
    {"synthesize": "synthesize", "end": END},
)
main_builder.add_conditional_edges(
    "rag_agent",
    check_if_synthesis_needed,
    {"synthesize": "synthesize", "end": END},
)
main_builder.add_edge("reasoning", END)
main_builder.add_edge("synthesize", END)

graph = main_builder.compile()
