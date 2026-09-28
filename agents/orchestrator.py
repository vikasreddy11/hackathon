"""
agents/orchestrator.py
======================
The core of Agentic GraphRAG — the autonomous investigation loop.

Orchestrator Rules & Architecture:
-----------------------------------
1. Avoid repeated searches: Do not perform nearly identical searches. If 2 searches
   target the same semantic area without improvement, change strategy.
2. Search for missing information: Use missing_information from evaluate_evidence()
   to target specific gaps rather than rephrasing the original question.
3. Re-evaluate after meaningful new evidence: Call evaluate_evidence() when new evidence
   is collected or strategy changes.
4. Stop early: If evidence is sufficient, stop with stopping_reason = "sufficient_evidence".
5. Handle weak evidence honestly: If stopped without full readiness, answer only what the
   evidence supports and explain missing information.
6. Separate retrieved evidence from cited evidence.
7. Use graph facts with source provenance.
8. Choose tools dynamically based on the question (conceptual vs multi-hop/relational).
9. Track full state (searches, entities, graph facts, scores, gaps, budget, strategy changes).
10. Respect budgets: stop if token budget or max steps are reached.
11. Stop on no progress: If 2 consecutive actions add no meaningful evidence, stop.
12. Final grounded answer: Ground strictly on evidence without inventing unsupported claims.
"""

from __future__ import annotations

import json
import re
from typing import Any

from rag.llm_client import call_llm
from agents.state import AgentState, ToolResult
from agents.tools import (
    TOOL_DESCRIPTIONS,
    TOOL_REGISTRY,
    document_search,
    entity_search,
    evaluate_evidence,
    graph_search,
    vector_search,
)
from graphrag.context_builder import build_graph_context


# ---------------------------------------------------------------------------
# System prompt for the orchestrator LLM
# ---------------------------------------------------------------------------

_ORCHESTRATOR_SYSTEM_PROMPT = """\
You are an Agentic GraphRAG orchestrator.
Your goal is to answer the user's question with the minimum sufficient grounded evidence. Dynamically choose tools; do not follow a fixed sequence.

Available tools:
- vector_search(query, top_k): semantic search over document chunks
- entity_search(query): find relevant graph entities
- graph_search(entity_ids): explore relationships/multi-hop connections
- document_search(chunk_ids): retrieve source text supporting graph facts
- evaluate_evidence(): return score, ready, missing_information, reasoning

Rules:
1. Avoid repeated searches: If 2 searches target the same semantic area without improvement, change strategy.
2. Search for missing information: After evaluate_evidence(), use missing_information to target specific gaps.
3. Re-evaluate after meaningful new evidence: After useful retrieval or strategy change, call evaluate_evidence().
4. Stop early: If evidence is sufficient (score >= threshold or ready=True), stop.
5. Handle weak evidence honestly: If stopped without full readiness, answer only supported facts and state gaps.
6. Separate retrieved vs cited evidence.
7. Choose tools based on question type:
   - Simple conceptual: vector_search -> evaluate_evidence -> answer
   - Relational/multi-hop: entity_search -> graph_search -> document_search -> evaluate
8. You ALWAYS respond with valid JSON only.
"""

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_TOOL_SELECTION_PROMPT = """\
You are investigating the following question:

QUESTION: {question}

TOOLS AVAILABLE:
{tool_descriptions}

INVESTIGATION STATUS:
- Steps taken so far: {step_count} (Max allowed: {max_steps})
- Tokens used: ~{tokens_used} (Budget: {token_budget})
- Known Entities: {entities_found}
- Current Evidence Score: {evidence_score}/10
- Ready Status: {ready_status}
- Missing Information to Target: {missing_info}

STEPS TAKEN SO FAR:
{steps_taken}

EVIDENCE COLLECTED SO FAR:
{evidence_summary}

YOUR TASK:
Choose the SINGLE best tool to call next to make progress toward answering the question.
If missing information is specified above, target those specific gaps rather than repeating broad searches.

Respond with ONLY this JSON (no other text):
{{
  "tool": "<tool_name>",
  "args": {{<tool-specific arguments as key-value pairs>}},
  "reasoning": "<one sentence explaining why this tool and these arguments>",
  "strategy_change": <true/false whether this switches retrieval strategy>
}}

Rules:
- "tool" must be one of: {tool_names}
- For vector_search: args must include "query" (string) and optionally "top_k" (integer)
- For entity_search: args must include "query" (string)
- For graph_search: args must include "entity_ids" (list of strings)
- For document_search: args must include "chunk_ids" (list of strings)
- For evaluate_evidence: args must be empty {{}}
- Do NOT repeat an identical tool call already made.
"""

_FINAL_ANSWER_PROMPT = """\
You are a knowledgeable research assistant. Answer the following question using ONLY
the evidence provided. Be comprehensive, precise, and cite the evidence (using chunk IDs).

QUESTION: {question}

EVIDENCE SUFFICIENCY STATUS:
Ready: {ready}
Evidence Score: {score}/10
Missing Information (if any): {missing_info}

RETRIEVED EVIDENCE:
{evidence_text}

KNOWLEDGE GRAPH CONTEXT:
{graph_context}

Instructions:
1. Synthesize a complete, accurate answer using ONLY the evidence above.
2. Reference specific evidence chunks when making factual claims (e.g. "[doc1_llm_overview__chunk_0000]").
3. If Ready is False or evidence is partial, answer only what the evidence supports and explicitly mention what could not be verified.
4. Do NOT invent facts or hallucinate citations.
5. Be clear and well-structured.

Answer:
"""


# ---------------------------------------------------------------------------
# AgentOrchestrator
# ---------------------------------------------------------------------------

class AgentOrchestrator:
    """
    Runs the agentic investigation loop.

    Parameters
    ----------
    max_steps : int
        Maximum number of tool calls before forcing a stop.
    token_budget : int
        Approximate token budget. Stop if exceeded.
    evidence_threshold : int
        Score (0-10) from evaluate_evidence that is considered "sufficient".
    backend : str
        LLM backend to use for tool selection and answer generation.
    model : str | None
        LLM model override.
    verbose : bool
        If True, print step-by-step progress to console.
    """

    def __init__(
        self,
        max_steps: int = 8,
        token_budget: int = 4000,
        evidence_threshold: int = 7,
        backend: str = "gemini",
        model: str | None = None,
        verbose: bool = True,
    ):
        self.max_steps = max_steps
        self.token_budget = token_budget
        self.evidence_threshold = evidence_threshold
        self.backend = backend
        self.model = model
        self.verbose = verbose

        self._tool_names = list(TOOL_REGISTRY.keys())
        self._tool_desc_block = "\n".join(
            f"  - {name}: {desc}"
            for name, desc in TOOL_DESCRIPTIONS.items()
        )

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def run(self, question: str) -> AgentState:
        """
        Run the full investigation loop for a question.
        """
        state = AgentState(question=question)
        seen_calls: set[tuple[str, str]] = set()
        consecutive_no_progress = 0
        prev_evidence_count = 0
        prev_entity_count = 0
        last_tool_type = None

        self._log(f"\n{'='*60}")
        self._log(f"AGENTIC INVESTIGATION STARTED")
        self._log(f"Question: {question}")
        self._log(f"{'='*60}")

        # ----------------------------------------------------------------
        # Investigation loop
        # ----------------------------------------------------------------
        while True:
            # -- Stopping criteria 1: Max steps --
            if state.step_count >= self.max_steps:
                state.stopped_reason = "max_steps"
                break

            # -- Stopping criteria 2: Token budget --
            if state.total_tokens_approx >= self.token_budget:
                state.stopped_reason = "budget_exceeded"
                break

            # -- Select next tool --
            tool_name, tool_args, reasoning, strategy_flag = self._select_next_tool(state)

            # Strategy change detection
            current_tool_type = "graph" if "graph" in tool_name or "entity" in tool_name else ("eval" if tool_name == "evaluate_evidence" else "vector")
            is_strategy_change = strategy_flag or (last_tool_type is not None and current_tool_type != last_tool_type and current_tool_type != "eval")
            if is_strategy_change:
                state.strategy_changes += 1
            last_tool_type = current_tool_type

            # Dedup guard
            call_key_arg = str(tool_args.get("query") or tool_args.get("entity_ids") or tool_args.get("chunk_ids") or "")
            call_signature = (tool_name, call_key_arg)
            if call_signature in seen_calls and tool_name != "evaluate_evidence":
                tool_name = "evaluate_evidence"
                tool_args = {}
                reasoning = "Forced evaluation after duplicate tool call detected"
                is_strategy_change = True
            seen_calls.add(call_signature)

            # Execute tool
            tool_result = self._execute_tool(
                tool_name=tool_name,
                tool_args=tool_args,
                reasoning=reasoning,
                state=state,
            )

            # Record in state
            state.add_tool_result(tool_result)

            # Check new evidence added this step
            curr_evidence_count = len(state.evidence)
            curr_entity_count = len(state.entities_found)
            new_chunks = curr_evidence_count - prev_evidence_count
            new_entities = curr_entity_count - prev_entity_count
            new_evidence_str = f"{new_chunks} new chunk(s), {new_entities} new entity(s)" if (new_chunks or new_entities) else "None"

            evidence_score_str = "N/A"

            # Process evaluation result
            if tool_name == "evaluate_evidence":
                eval_result = tool_result.outputs
                score = eval_result.get("score", 0)
                ready = eval_result.get("ready", False)
                missing = eval_result.get("missing", [])

                state.latest_score = score
                state.latest_ready = ready
                state.latest_missing = missing
                evidence_score_str = f"{score}/10 (Ready: {ready})"

                if ready or score >= self.evidence_threshold:
                    state.stopped_reason = "sufficient_evidence"
                    self._log_step(
                        step_n=state.step_count,
                        tool=tool_name,
                        reason=reasoning,
                        inputs=tool_args,
                        new_evidence=new_evidence_str,
                        score=evidence_score_str,
                        strategy_changed="Yes" if is_strategy_change else "No",
                        tokens=tool_result.tokens_used,
                    )
                    break
            elif state.latest_score > 0:
                evidence_score_str = f"{state.latest_score}/10"

            self._log_step(
                step_n=state.step_count,
                tool=tool_name,
                reason=reasoning,
                inputs=tool_args,
                new_evidence=new_evidence_str,
                score=evidence_score_str,
                strategy_changed="Yes" if is_strategy_change else "No",
                tokens=tool_result.tokens_used,
            )

            # Check for no-progress stopping condition (Rule 11)
            if new_chunks == 0 and new_entities == 0 and tool_name != "evaluate_evidence":
                consecutive_no_progress += 1
            else:
                consecutive_no_progress = 0

            if consecutive_no_progress >= 2:
                state.stopped_reason = "no_progress"
                break

            prev_evidence_count = curr_evidence_count
            prev_entity_count = curr_entity_count

        # Set default stop reason if not set
        if not state.stopped_reason:
            state.stopped_reason = "sufficient_evidence" if state.latest_ready else "max_steps"

        # Print STOP summary
        self._log_stop_summary(state)

        # ----------------------------------------------------------------
        # Generate final answer
        # ----------------------------------------------------------------
        self._log(f"\n{'='*60}")
        self._log(f"GENERATING FINAL ANSWER...")
        state.final_answer = self._generate_final_answer(state)
        state.final_sources = self._extract_cited_sources(state)
        self._log(f"Answer generated. Sources: {len(state.final_sources)}")
        self._log(f"{'='*60}")

        return state

    # -----------------------------------------------------------------------
    # Tool selection
    # -----------------------------------------------------------------------

    def _select_next_tool(
        self, state: AgentState
    ) -> tuple[str, dict[str, Any], str, bool]:
        """
        Ask LLM which tool to call next and with what arguments.
        """
        missing_str = ", ".join(state.latest_missing) if state.latest_missing else "None identified yet"
        prompt = _TOOL_SELECTION_PROMPT.format(
            question=state.question,
            tool_descriptions=self._tool_desc_block,
            step_count=state.step_count,
            max_steps=self.max_steps,
            tokens_used=state.total_tokens_approx,
            token_budget=self.token_budget,
            entities_found=", ".join(state.entities_found) if state.entities_found else "None yet",
            evidence_score=state.latest_score,
            ready_status=state.latest_ready,
            missing_info=missing_str,
            steps_taken=state.get_trace_summary() or "None yet",
            evidence_summary=state.get_evidence_summary(),
            tool_names=", ".join(self._tool_names),
        )

        try:
            raw = call_llm(
                prompt=prompt,
                system_prompt=_ORCHESTRATOR_SYSTEM_PROMPT,
                backend=self.backend,
                model=self.model,
                temperature=0.0,
                max_tokens=256,
            )
            tool_name, tool_args, reasoning, strategy_flag = _parse_tool_selection(raw)
        except Exception as exc:
            self._log(f"[WARNING] Tool selection failed: {exc}. Falling back to vector_search.")
            tool_name = "vector_search"
            tool_args = {"query": state.question, "top_k": 3}
            reasoning = f"Fallback due to selection error: {exc}"
            strategy_flag = False

        if tool_name not in TOOL_REGISTRY:
            tool_name = "vector_search"
            tool_args = {"query": state.question, "top_k": 3}
            reasoning = "Corrected unknown tool to vector_search."
            strategy_flag = False

        return tool_name, tool_args, reasoning, strategy_flag

    # -----------------------------------------------------------------------
    # Tool execution
    # -----------------------------------------------------------------------

    def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        reasoning: str,
        state: AgentState,
    ) -> ToolResult:
        """Dispatch to the appropriate tool function."""
        try:
            if tool_name == "vector_search":
                return vector_search(
                    query=tool_args.get("query", state.question),
                    top_k=int(tool_args.get("top_k", 3)),
                    reasoning=reasoning,
                )
            elif tool_name == "entity_search":
                return entity_search(
                    query=tool_args.get("query", state.question),
                    known_entities=tool_args.get("known_entities"),
                    max_entities=int(tool_args.get("max_entities", 10)),
                    max_chunks=int(tool_args.get("max_chunks", 3)),
                    reasoning=reasoning,
                )
            elif tool_name == "graph_search":
                entity_ids = tool_args.get("entity_ids", state.entities_found[:3])
                if not entity_ids:
                    entity_ids = state.entities_found[:3] or [state.question.split()[0].lower()]
                return graph_search(
                    entity_ids=entity_ids,
                    max_entities=int(tool_args.get("max_entities", 15)),
                    max_chunks=int(tool_args.get("max_chunks", 4)),
                    reasoning=reasoning,
                )
            elif tool_name == "document_search":
                chunk_ids = tool_args.get("chunk_ids", state.subgraph_data.get("chunk_ids", [])[:3])
                if not chunk_ids:
                    return ToolResult(
                        tool_name="document_search",
                        inputs={"chunk_ids": []},
                        outputs=[],
                        tokens_used=0,
                        reasoning="Skipped: no chunk_ids available yet.",
                    )
                return document_search(chunk_ids=chunk_ids, reasoning=reasoning)
            elif tool_name == "evaluate_evidence":
                return evaluate_evidence(
                    question=state.question,
                    evidence=state.evidence,
                    backend=self.backend,
                    model=self.model,
                    reasoning=reasoning,
                )
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

        except Exception as exc:
            self._log(f"[ERROR] Tool '{tool_name}' raised: {exc}. Returning empty result.")
            return ToolResult(
                tool_name=tool_name,
                inputs=tool_args,
                outputs=[],
                tokens_used=0,
                reasoning=f"Tool execution failed: {exc}",
            )

    # -----------------------------------------------------------------------
    # Final answer generation
    # -----------------------------------------------------------------------

    def _generate_final_answer(self, state: AgentState) -> str:
        """Generate the final answer using all accumulated evidence."""
        evidence_lines = []
        for i, chunk in enumerate(state.evidence[:8], 1):
            cid = chunk.get("chunk_id", f"chunk_{i}")
            doc = chunk.get("doc_id", "unknown")
            text = chunk.get("text", "").strip()[:600]
            evidence_lines.append(f"[{cid} | {doc}]\n{text}")
        evidence_text = "\n\n".join(evidence_lines) if evidence_lines else "No evidence retrieved."

        if state.subgraph_data.get("entities") or state.subgraph_data.get("relationships"):
            graph_context = build_graph_context(state.subgraph_data)
        else:
            graph_context = "(No knowledge graph entities retrieved)"

        missing_str = ", ".join(state.latest_missing) if state.latest_missing else "None"

        prompt = _FINAL_ANSWER_PROMPT.format(
            question=state.question,
            ready=state.latest_ready,
            score=state.latest_score,
            missing_info=missing_str,
            evidence_text=evidence_text,
            graph_context=graph_context,
        )

        try:
            answer = call_llm(
                prompt=prompt,
                system_prompt=(
                    "You are a knowledgeable research assistant. Provide clear, "
                    "accurate, well-cited answers based strictly on the provided evidence."
                ),
                backend=self.backend,
                model=self.model,
                temperature=0.0,
                max_tokens=1024,
            )
            state.total_tokens_approx += len(prompt.split()) + len(answer.split())
            return answer
        except Exception as exc:
            return f"[Answer generation failed: {exc}] Evidence gathered: {len(state.evidence)} chunks."

    # -----------------------------------------------------------------------
    # Source extraction & separation of retrieved vs cited evidence (Rule 6)
    # -----------------------------------------------------------------------

    def _extract_cited_sources(self, state: AgentState) -> list[dict[str, Any]]:
        """
        Separate retrieved evidence from cited evidence.
        Only chunks directly mentioned in the final answer (or all retrieved if none explicitly cited)
        are returned as final cited sources.
        """
        answer = state.final_answer
        cited = []
        seen = set()

        for chunk in state.evidence:
            cid = chunk.get("chunk_id", "")
            if cid and cid in answer and cid not in seen:
                seen.add(cid)
                cited.append({
                    "chunk_id": cid,
                    "doc_id": chunk.get("doc_id", ""),
                    "source": chunk.get("source", ""),
                    "score": chunk.get("score", 0.0),
                })

        # Fallback to deduplicated retrieved evidence if LLM did not include explicit inline brackets
        if not cited:
            for chunk in state.evidence:
                cid = chunk.get("chunk_id", "")
                if cid not in seen:
                    seen.add(cid)
                    cited.append({
                        "chunk_id": cid,
                        "doc_id": chunk.get("doc_id", ""),
                        "source": chunk.get("source", ""),
                        "score": chunk.get("score", 0.0),
                    })
        return cited

    # -----------------------------------------------------------------------
    # Logging formatting
    # -----------------------------------------------------------------------

    def _log_step(
        self,
        step_n: int,
        tool: str,
        reason: str,
        inputs: dict[str, Any],
        new_evidence: str,
        score: str,
        strategy_changed: str,
        tokens: int,
    ) -> None:
        if not self.verbose:
            return
        print(f"\nStep {step_n}")
        print(f"Tool: [{tool}]")
        print(f"Reason: {reason}")
        print(f"Inputs: {inputs}")
        print(f"New evidence: {new_evidence}")
        print(f"Evidence score: {score}")
        print(f"Strategy changed: {strategy_changed}")
        print(f"Tokens used: ~{tokens}")

    def _log_stop_summary(self, state: AgentState) -> None:
        if not self.verbose:
            return
        tools_used = [r.tool_name for r in state.trace]
        missing_str = ", ".join(state.latest_missing) if state.latest_missing else "None"

        print(f"\n{'='*60}")
        print("STOP")
        print(f"Reason: {state.stopped_reason}")
        print(f"Evidence score: {state.latest_score}/10")
        print(f"Ready: {state.latest_ready}")
        print(f"Missing information: {missing_str}")
        print(f"Steps: {state.step_count}")
        print(f"Tools used: {tools_used}")
        print(f"Strategy changes: {state.strategy_changes}")
        print(f"Retrieved evidence: {len(state.evidence)} chunks")
        print(f"Cited evidence: {len(state.final_sources)} sources")
        print(f"{'='*60}")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def _parse_tool_selection(text: str) -> tuple[str, dict[str, Any], str, bool]:
    """
    Parse the LLM's tool-selection JSON response.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"No valid JSON in tool selection response: {text[:300]}")

    tool_name = str(data.get("tool", "vector_search"))
    tool_args = dict(data.get("args", {}))
    reasoning = str(data.get("reasoning", ""))
    strategy_change = bool(data.get("strategy_change", False))
    return tool_name, tool_args, reasoning, strategy_change
