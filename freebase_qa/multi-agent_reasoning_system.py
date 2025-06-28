import os
import uuid
import faiss
from langchain_openai import AzureChatOpenAI
from sentence_transformers import SentenceTransformer
from langchain_community.docstore.in_memory import InMemoryDocstore
from typing_extensions import TypedDict
from typing import Annotated, List, Dict
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps
# from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import ast
from core.freebase_client import FreebaseClient
from core.semantic_search import SemanticSearch
from utils.text_utils import TextUtils
import numpy as np

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] ="true"
os.environ["LANGSMITH_PROJECT"] ="graph-rag-system"
thread_id = uuid.uuid4() 

# Initialize clients and utilities
fb_client = FreebaseClient()
text_utils = TextUtils()
semantic_searcher = SemanticSearch()

# Models initialization
llm = AzureChatOpenAI(
    azure_endpoint="https://73476-m9mi3n1l-eastus2.cognitiveservices.azure.com/",
    azure_deployment="gpt-4.1",
    api_version="2024-02-15-preview",
    temperature=0,
    max_tokens=None,
)
sbert = SentenceTransformer("all-MiniLM-L6-v2")

# Adjustable parameters
args = {
    "mid_num": 3,
    "width": 3,
    "depth": 3,
    "num_retain_entity": 10,
    "keyword_num": 5,
    "relation_num": 5,
    "prune_tools": "llm",
    "remove_unnecessary_rel": True
}
config = {"configurable": {"thread_id": thread_id}} 

@tool
def get_topic_entities(question, words):
    """Get topic entities from Freebase."""
    keywords = set()
    for w in words:
        keywords.add(w)
        query_embedding = sbert.encode([w], normalize_embeddings=True, show_progress_bar=False)
        _, indices = semantic_searcher.faiss_index.search(query_embedding, args['keyword_num'])
        keywords.update(semantic_searcher.names[idx] for idx in indices[0])
    keywords = list(keywords)

    mid_scores = []
    for keyword in keywords:
        eids = fb_client.get_entity_id(keyword)
        for eid in eids:
            predicates = fb_client.get_all_relations(eid)
            if not predicates:
                continue

            # Deduplication (optional)
            predicates = list(set(predicates))

            # Calculate the similarity between each predicate and the problem
            predicate_embeddings = sbert.encode(predicates, normalize_embeddings=True, show_progress_bar=False)
            dim = predicate_embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(predicate_embeddings)
            query_embedding = sbert.encode([question], normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
            top_k = min(args['relation_num'], len(predicates))
            scores, indices = index.search(query_embedding, top_k)
            top_scores = scores[0].tolist()
            avg_top_score = sum(top_scores) / len(top_scores) if top_scores else 0
            mid_scores.append((eid, keyword, avg_top_score))

    # Select the mid with top-3 similarity
    top_mids = sorted(mid_scores, key=lambda x: x[2], reverse=True)[:args['mid_num']]

    topic_entities = {mid: name for mid, name, _ in top_mids}

    return topic_entities


entity_recognition_agent = create_react_agent(
    model=llm,
    tools=[get_topic_entities],
    name="entity_recognition_expert",
    prompt="You are a entity recognition expert. Given a question, suppose you don't know the answer to the question and need to retrieve it from the knowledge graph. You have to determine the entities that might exist in the knowledge graph, and then return the corresponding ids of these entities by calling the tool.",
)

result = entity_recognition_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "what was shakespeare's wife called?"
            }
        ]
    },
    config=config
)

for message in result["messages"]:  
   message.pretty_print()










# def add(a: float, b: float) -> float:
#     """Add two numbers."""
#     return a + b

# def multiply(a: float, b: float) -> float:
#     """Multiply two numbers."""
#     return a * b

# def web_search(query: str) -> str:
#     """Search the web for information."""
#     return (
#         "Here are the headcounts for each of the FAANG companies in 2024:\n"
#         "1. **Facebook (Meta)**: 67,317 employees.\n"
#         "2. **Apple**: 164,000 employees.\n"
#         "3. **Amazon**: 1,551,000 employees.\n"
#         "4. **Netflix**: 14,000 employees.\n"
#         "5. **Google (Alphabet)**: 181,269 employees."
#     )

# graph_research_agent = create_react_agent(
#     model=model,
#     tools=[web_search],
#     name="graph_research_expert",
#     prompt="You are a world class researcher with access to web search. Do not do any math."
# )

# # Create supervisor workflow
# workflow = create_supervisor(
#     [entity_recognition_agent, graph_research_agent],
#     model=model,
#     prompt=(
#         "You are a team supervisor managing a research expert and a math expert. "
#         "For current events, use research_agent. "
#         "For math problems, use math_agent."
#     )
# )

# # Compile and run
# app = workflow.compile()
# result = app.invoke({
#     "messages": [
#         {
#             "role": "user",
#             "content": "what's the combined headcount of the FAANG companies in 2024?"
#         }
#     ]
# })









