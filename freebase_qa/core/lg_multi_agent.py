from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
from argparse import Namespace
from config.settings import FINISH_ID
from concurrent.futures import ThreadPoolExecutor
from langgraph.graph import StateGraph, END
from core.llm_handler import LLMHandler
from core.freebase_client import FreebaseClient
from core.reasoning_engine import ReasoningEngine
from core.semantic_search import SemanticSearch
from utils.logging_utils import logger
from copy import deepcopy
import json


class DiscussionState(TypedDict):
    args: Namespace
    question: str
    candidate_entities: Dict[int, Dict[str, str]]  # agent_id -> {mid: name}
    pre_heads: Dict[int, List[int]]               # agent_id -> pre_heads
    pre_relations: Dict[int, List[str]]           # agent_id -> pre_relations
    active_agents: List[int]
    current_depth: int
    reasoning_chains: Dict[int, List[List[str]]]
    knowledge: Dict[int, List[str]]
    whether_stop: bool
    final_answer: Optional[str]
    stop_reason: Optional[str]
    reasoning_log: List[str]
    output_file: str


class ReasoningAgent:
    def __init__(self, agent_id: int, fb: FreebaseClient, engine: ReasoningEngine):
        self.agent_id = agent_id
        self.fb = fb
        self.engine = engine

    def run(self, state: DiscussionState) -> Dict[str, Any]:
        if self.agent_id not in state.get("active_agents", []):
            return {"reasoning_log": [f"Agent {self.agent_id} is inactive"]}

        logger.info(f"[Agent {self.agent_id}] Start reasoning at depth {state['current_depth']}")

        topic_entity = state.get("candidate_entities", {}).get(self.agent_id, {})
        pre_heads = state.get("pre_heads", {}).get(self.agent_id, [-1] * len(topic_entity))
        pre_relations = state.get("pre_relations", {}).get(self.agent_id, [])

        current_relations = self.engine._search_and_prune_relations(
            state["question"], topic_entity, pre_relations, pre_heads, state["args"])

        if not current_relations:
            return {"reasoning_log": [f"Agent {self.agent_id}: no valid relations found"], "deactivated_agent": self.agent_id}

        candidates = self.engine._collect_and_score_entities(
            state["question"], current_relations, state["args"])

        if not candidates:
            return {"reasoning_log": [f"Agent {self.agent_id}: no valid entities found"], "deactivated_agent": self.agent_id}

        chain_of_entities, candidate_entities, new_pre_relations, new_pre_heads = self.engine.triples_prune(state["question"], *candidates)

        # flag, chain_of_entities, entities_id, new_pre_relations, new_pre_heads = self.engine.entity_prune(
        #     *candidates, state["args"])

        # if not flag:
        #     return {"reasoning_log": [f"Agent {self.agent_id}: pruning failed"], "deactivated_agent": self.agent_id}

        # candidate_entities = {
        #     eid: self.fb.get_entity_info(eid) for eid in entities_id if eid != FINISH_ID
        # }

        # is_relevant, summary = self.engine.is_relevant_to_question(state["question"], chain_of_entities, state["args"])
        # if not is_relevant:
        #     return {"reasoning_log": [f"Agent {self.agent_id}: triples are irrelevant. Stopped."], "deactivated_agent": self.agent_id}

        return {
            "candidate_entities": {self.agent_id: candidate_entities},
            "pre_heads": {self.agent_id: new_pre_heads},
            "pre_relations": {self.agent_id: new_pre_relations},
            "reasoning_chains": {self.agent_id: [chain_of_entities]},
            "knowledge": {self.agent_id: []},
            "reasoning_log": [f"Agent {self.agent_id} completed."]
        }


class KnowledgeGraphReasoningSystem:
    def __init__(self, llm_handler: LLMHandler, freebase_client: FreebaseClient, ss: SemanticSearch, agent_count: int = 3):
        self.llm = llm_handler
        self.fb = freebase_client
        self.ss = ss
        self.reason_engine = ReasoningEngine(self.fb, self.llm, self.ss)
        self.executor = ThreadPoolExecutor(max_workers=agent_count)
        self.agent_pool = {
            i + 1: ReasoningAgent(i + 1, self.fb, self.reason_engine)
            for i in range(agent_count)
        }
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        workflow = StateGraph(DiscussionState)

        workflow.add_node("filter", self.filter)
        for agent_id in self.agent_pool:
            workflow.add_node(f"agent_{agent_id}", self.run_agent_factory(agent_id))

        workflow.add_node("parallel_agents", self.parallel_agents_node)
        workflow.add_node("supervisor", self.supervisor)
        workflow.set_entry_point("filter")
        workflow.add_edge("filter", "parallel_agents")

        def route_supervisor(state: DiscussionState):
            if state["whether_stop"] or state["current_depth"] > state["args"].depth:
                return END
            return "parallel_agents"

        workflow.add_edge("parallel_agents", "supervisor")
        workflow.add_conditional_edges("supervisor", route_supervisor, {END: END, "parallel_agents": "parallel_agents"})
        return workflow.compile()

    def filter(self, state: DiscussionState):
        question = state["question"]
        mid_scores = []

        top_k = state["args"].relation_num  # 可调参数：取每个mid的Top-K关系相似度来平均

        # 1. 获取关键词对应的实体及其所有关系
        keywords = self.reason_engine.retrieve_keyword(question, state["args"])
        for k in keywords:
            eids = self.fb.get_entity_id(k)
            for eid in eids:
                predicates = self.fb.get_all_relations(eid)
                if not predicates:
                    continue

                # 去重（可选）
                predicates = list(set(predicates))

                # 2. 计算每个 predicate 与问题的相似度
                sim_scores = self.llm.compute_similarity_batch(question, predicates)

                # 取 Top-K 平均（代替所有平均）
                top_scores = sorted(sim_scores, reverse=True)[:top_k]
                avg_top_score = sum(top_scores) / len(top_scores)
                mid_scores.append((eid, k, avg_top_score))

        # 3. 选出 top-3 相似度的 mid
        top_mids = sorted(mid_scores, key=lambda x: x[2], reverse=True)[:state["args"].agent_count]

        # 4. 构造 topic_entities 格式：{agent_id: {mid: name}}
        topic_entities = {i + 1: {mid: name} for i, (mid, name, _) in enumerate(top_mids)}

        pre_heads = {i + 1: [-1] for i in range(len(topic_entities))}
        pre_relations = {i + 1: [] for i in range(len(topic_entities))}

        return {
            "candidate_entities": topic_entities,
            "pre_heads": pre_heads,
            "pre_relations": pre_relations,
            "current_depth": 1,
            "active_agents": list(self.agent_pool.keys()),
            "reasoning_log": ["Filter agent selected top topic entities (using SBERT Top-K similarity)."]
        }


    def run_agent_factory(self, agent_id: int):
        agent = self.agent_pool[agent_id]
        return lambda state: agent.run(state)

    def parallel_agents_node(self, state: DiscussionState):
        futures, updates = [], []
        for agent_id in state["active_agents"]:
            agent_func = self.run_agent_factory(agent_id)
            futures.append((agent_id, self.executor.submit(agent_func, deepcopy(state))))

        for agent_id, future in futures:
            try:
                update = future.result(timeout=500)
                if update:
                    updates.append(update)
            except Exception as e:
                logger.error(f"Agent {agent_id} execution error: {e}")
                raise

        return self.merge_agent_updates(updates, state)

    def merge_agent_updates(self, updates: List[Dict[str, Any]], state: DiscussionState):
        merged = {
            "reasoning_log": state.get("reasoning_log", []),
            "reasoning_chains": state.get("reasoning_chains", {}),
            "knowledge": state.get("knowledge", {}),
            "candidate_entities": {},  # will be overwritten
            "pre_heads": {},           # will be overwritten
            "pre_relations": {},       # will be overwritten
        }
        deactivated = set()

        for up in updates:
            merged["reasoning_log"].extend(up.get("reasoning_log", []))

            if up.get("deactivated_agent"):
                deactivated.add(up["deactivated_agent"])

            # 覆盖更新的字段
            for key in ["candidate_entities", "pre_heads", "pre_relations"]:
                if key in up:
                    merged[key].update(up[key])  # 直接覆盖（针对 agent_id）

            # 累加更新的字段
            for key in ["reasoning_chains", "knowledge"]:
                for agent_id, value in up.get(key, {}).items():
                    merged[key].setdefault(agent_id, []).extend(value)

        merged["active_agents"] = [a for a in state["active_agents"] if a not in deactivated]
        return merged


    def supervisor(self, state: DiscussionState):
        all_chains = [chain for chains in state.get("reasoning_chains", {}).values() for chain in chains]
        # combined_summary = "\n\n".join(
        #     " ".join(summaries) for summaries in state.get("knowledge", {}).values()
        # )

        # 1. 使用三元组作为外部知识
        stop, response = self.reason_engine.reasoning(state["question"], all_chains, state["args"])
        # 2. 使用文本化的三元组作为外部知识
        # stop, response = self.reason_engine.reasoning_with_summary(state["question"], combined_summary, state["args"])
        if stop:
            return self._finalize_state(state, "answer_found", response, state.get("reasoning_chains", {}))

        if not state.get("candidate_entities"):
            return self._finalize_state(state, "no_active_agents", self.reason_engine.generate_without_explored_paths(state["question"], state["args"]), {})

        if state["current_depth"] == state["args"].depth:
            return self._finalize_state(state, "max_depth_reached", self.reason_engine.generate_without_explored_paths(state["question"], state["args"]), {})

        return {
            **state,
            "current_depth": state["current_depth"] + 1,
            "active_agents": state["active_agents"],
            "reasoning_log": state.get("reasoning_log", []) + ["Supervisor decided to continue reasoning."]
        }

    def _finalize_state(self, state: DiscussionState, reason: str, answer: str, chains: Dict):
        logger.info(f"Finalizing with reason: {reason}, answer: {answer[:60]}...")
        self.reason_engine.save_results(state["question"], answer, chains, file_name=state["output_file"])
        return {
            **state,
            "final_answer": answer,
            "stop_reason": reason,
            "whether_stop": True,
            "reasoning_log": state.get("reasoning_log", []) + [f"Supervisor stopped due to {reason}."]
        }
