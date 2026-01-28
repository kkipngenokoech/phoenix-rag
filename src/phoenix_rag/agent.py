"""
Phoenix Agent - Main orchestrator for the RAG-enabled refactoring agent.

Implements a ReAct-style reasoning loop that:
1. Observes the user request
2. Reasons about what information/tools are needed
3. Acts by calling appropriate tools (retrieval or analysis)
4. Verifies the response for groundedness
5. Returns verified response to user
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from phoenix_rag.config import PhoenixConfig, config as default_config
from phoenix_rag.retrieval import RetrievalModule
from phoenix_rag.tools import ToolRegistry
from phoenix_rag.verification import VerificationModule

logger = logging.getLogger(__name__)


class AgentAction(Enum):
    """Actions the agent can take."""
    RETRIEVE = "retrieve"      # Search knowledge base
    ANALYZE = "analyze"        # Analyze code
    RESPOND = "respond"        # Generate final response
    CLARIFY = "clarify"        # Ask for clarification


@dataclass
class AgentStep:
    """A single step in the agent's reasoning process."""
    step_number: int
    thought: str
    action: AgentAction
    action_input: dict
    observation: str
    tool_used: Optional[str] = None


@dataclass
class AgentTrace:
    """Complete trace of agent execution for logging/debugging."""
    query: str
    steps: list[AgentStep] = field(default_factory=list)
    final_response: str = ""
    groundedness_score: float = 0.0
    total_iterations: int = 0
    tools_used: list[str] = field(default_factory=list)
    verification_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert trace to dictionary for logging."""
        return {
            "query": self.query,
            "steps": [
                {
                    "step": s.step_number,
                    "thought": s.thought,
                    "action": s.action.value,
                    "tool_used": s.tool_used,
                    "observation_preview": s.observation[:200] + "..." if len(s.observation) > 200 else s.observation,
                }
                for s in self.steps
            ],
            "final_response": self.final_response[:500] + "..." if len(self.final_response) > 500 else self.final_response,
            "groundedness_score": self.groundedness_score,
            "total_iterations": self.total_iterations,
            "tools_used": self.tools_used,
            "verification": self.verification_metadata,
        }


class PhoenixAgent:
    """
    Phoenix RAG Agent for code refactoring assistance.

    This agent implements a ReAct-style reasoning loop:
    - Thought: Reason about what to do
    - Action: Call a tool (retrieval, code analysis, etc.)
    - Observation: Process tool output
    - Repeat until ready to respond
    - Verify: Check response groundedness before returning

    The agent decides whether to:
    1. Retrieve knowledge from the RAG system
    2. Analyze provided code
    3. Generate a response based on gathered information
    """

    SYSTEM_PROMPT = """You are Phoenix, an AI assistant specialized in code refactoring.
You help developers improve code quality by identifying code smells, suggesting refactoring patterns,
and providing best practices.

You have access to a knowledge base of refactoring patterns, code smells, and best practices.
You also have tools to analyze code for structural issues.

When responding to queries:
1. First determine what information you need (code analysis, refactoring patterns, etc.)
2. Use the appropriate tools to gather information
3. Synthesize the information into a helpful response
4. Always ground your suggestions in established refactoring patterns

Be specific, practical, and cite the patterns/principles you're applying.
"""

    REACT_PROMPT = """Based on the user's query and available information, decide your next action.

User Query: {query}

Previous Steps:
{previous_steps}

Available Actions:
1. RETRIEVE: Search the knowledge base for refactoring patterns, best practices, or code smell remediation
2. ANALYZE: Analyze the provided code for structure, complexity, and code smells
3. RESPOND: Generate the final response based on gathered information
4. CLARIFY: Ask the user for clarification if the query is unclear

Available Tools:
- knowledge_retrieval: Search for refactoring knowledge (use with RETRIEVE action)
- code_analyzer: Analyze code structure and detect code smells (use with ANALYZE action)
- complexity_calculator: Calculate detailed code metrics (use with ANALYZE action)

Respond in this exact JSON format:
{{
    "thought": "Your reasoning about what to do next",
    "action": "RETRIEVE" | "ANALYZE" | "RESPOND" | "CLARIFY",
    "action_input": {{
        "tool": "tool_name" (if using a tool),
        "parameters": {{ tool parameters }},
        "response": "your response" (if action is RESPOND or CLARIFY)
    }}
}}
"""

    def __init__(
        self,
        config: Optional[PhoenixConfig] = None,
        retrieval_module: Optional[RetrievalModule] = None,
    ):
        """
        Initialize the Phoenix Agent.

        Args:
            config: Phoenix configuration
            retrieval_module: Pre-initialized retrieval module (optional)
        """
        self.config = config or default_config
        self.config.ensure_directories()

        # Initialize LLM
        self.llm = ChatAnthropic(
            model=self.config.llm.model,
            api_key=self.config.llm.api_key,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
        )

        # Initialize modules
        self.retrieval = retrieval_module or RetrievalModule(self.config)
        self.tools = ToolRegistry(self.retrieval)
        self.verification = VerificationModule(
            self.config,
            groundedness_threshold=self.config.agent.groundedness_threshold,
        )

        # Conversation state
        self.conversation_history: list = []
        self.current_trace: Optional[AgentTrace] = None
        self.retrieved_sources: list[dict] = []

        logger.info("PhoenixAgent initialized")

    def run(
        self,
        query: str,
        code: Optional[str] = None,
        max_iterations: Optional[int] = None,
    ) -> tuple[str, AgentTrace]:
        """
        Run the agent on a user query.

        Args:
            query: User's question or request
            code: Optional code snippet to analyze
            max_iterations: Maximum reasoning iterations (default from config)

        Returns:
            (response, trace) - The agent's response and execution trace
        """
        max_iterations = max_iterations or self.config.agent.max_iterations

        # Initialize trace
        self.current_trace = AgentTrace(query=query)
        self.retrieved_sources = []

        # If code is provided, add it to the query context
        if code:
            query = f"{query}\n\nCode to analyze:\n```python\n{code}\n```"

        logger.info(f"Agent processing query: {query[:100]}...")

        # ReAct loop
        previous_steps = []
        for iteration in range(max_iterations):
            step = self._react_step(query, previous_steps, iteration + 1)
            self.current_trace.steps.append(step)
            previous_steps.append(step)

            if step.action == AgentAction.RESPOND:
                # Ready to respond - verify and return
                response = step.action_input.get("response", "")
                verified_response = self._verify_response(response, query)
                self.current_trace.final_response = verified_response
                self.current_trace.total_iterations = iteration + 1
                break

            elif step.action == AgentAction.CLARIFY:
                # Need clarification
                clarification = step.action_input.get("response", "Could you please clarify your question?")
                self.current_trace.final_response = clarification
                self.current_trace.total_iterations = iteration + 1
                break

        else:
            # Max iterations reached
            logger.warning(f"Max iterations ({max_iterations}) reached")
            self.current_trace.final_response = self._generate_fallback_response(query, previous_steps)
            self.current_trace.total_iterations = max_iterations

        return self.current_trace.final_response, self.current_trace

    def _react_step(
        self,
        query: str,
        previous_steps: list[AgentStep],
        step_number: int,
    ) -> AgentStep:
        """Execute a single ReAct step."""
        # Format previous steps for context
        steps_text = self._format_previous_steps(previous_steps)

        # Get agent's decision
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=self.REACT_PROMPT.format(
                query=query,
                previous_steps=steps_text or "None yet",
            )),
        ]

        response = self.llm.invoke(messages)
        decision = self._parse_decision(response.content)

        # Execute the action
        observation = self._execute_action(decision)

        return AgentStep(
            step_number=step_number,
            thought=decision.get("thought", ""),
            action=AgentAction(decision.get("action", "respond").upper()),
            action_input=decision.get("action_input", {}),
            observation=observation,
            tool_used=decision.get("action_input", {}).get("tool"),
        )

    def _parse_decision(self, content: str) -> dict:
        """Parse the agent's decision from LLM output."""
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # Fallback: try to parse structured text
        return {
            "thought": content,
            "action": "respond",
            "action_input": {"response": content},
        }

    def _execute_action(self, decision: dict) -> str:
        """Execute the decided action and return observation."""
        action = decision.get("action", "").upper()
        action_input = decision.get("action_input", {})

        if action == "RETRIEVE":
            return self._execute_retrieval(action_input)
        elif action == "ANALYZE":
            return self._execute_analysis(action_input)
        elif action in ("RESPOND", "CLARIFY"):
            return action_input.get("response", "")
        else:
            return f"Unknown action: {action}"

    def _execute_retrieval(self, action_input: dict) -> str:
        """Execute knowledge retrieval."""
        tool_name = action_input.get("tool", "knowledge_retrieval")
        params = action_input.get("parameters", {})

        if tool_name == "knowledge_retrieval":
            query = params.get("query", "refactoring best practices")
            result = self.tools.execute(
                "knowledge_retrieval",
                query=query,
                doc_type=params.get("doc_type", "all"),
                num_results=params.get("num_results", 5),
            )

            if result.success:
                # Store retrieved sources for verification
                if isinstance(result.output, dict) and "results" in result.output:
                    self.retrieved_sources.extend([
                        {"content": r["content"], "metadata": {"source": r["source"]}}
                        for r in result.output["results"]
                    ])
                self.current_trace.tools_used.append("knowledge_retrieval")
                return result.to_string()
            else:
                return f"Retrieval failed: {result.error}"

        return "Invalid retrieval tool"

    def _execute_analysis(self, action_input: dict) -> str:
        """Execute code analysis."""
        tool_name = action_input.get("tool", "code_analyzer")
        params = action_input.get("parameters", {})

        code = params.get("code", "")
        if not code:
            return "No code provided for analysis"

        if tool_name == "code_analyzer":
            result = self.tools.execute(
                "code_analyzer",
                code=code,
                analysis_type=params.get("analysis_type", "full"),
            )
            self.current_trace.tools_used.append("code_analyzer")
        elif tool_name == "complexity_calculator":
            result = self.tools.execute(
                "complexity_calculator",
                code=code,
                metrics=params.get("metrics", ["all"]),
            )
            self.current_trace.tools_used.append("complexity_calculator")
        else:
            return f"Unknown analysis tool: {tool_name}"

        if result.success:
            return result.to_string()
        else:
            return f"Analysis failed: {result.error}"

    def _verify_response(self, response: str, original_query: str) -> str:
        """Verify and potentially correct the response."""
        if not self.retrieved_sources:
            # No sources to verify against
            self.current_trace.groundedness_score = 1.0
            self.current_trace.verification_metadata = {
                "note": "No sources retrieved for verification"
            }
            return response

        # Full verification with potential correction
        verified_response, confidence, metadata = self.verification.get_verified_response(
            response=response,
            sources=self.retrieved_sources,
            original_query=original_query,
        )

        self.current_trace.groundedness_score = metadata.get("groundedness_score", 0)
        self.current_trace.verification_metadata = metadata

        logger.info(f"Response verification: {metadata.get('groundedness_score', 0):.1%} groundedness")

        return verified_response

    def _generate_fallback_response(
        self,
        query: str,
        steps: list[AgentStep],
    ) -> str:
        """Generate a fallback response when max iterations reached."""
        # Gather any useful information from steps
        observations = "\n".join(s.observation[:500] for s in steps if s.observation)

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=f"""Based on the following information, provide a helpful response to the user's query.

Query: {query}

Gathered Information:
{observations}

Provide a concise, helpful response based on the available information. If information is incomplete, acknowledge this.
"""),
        ]

        response = self.llm.invoke(messages)
        return response.content

    def _format_previous_steps(self, steps: list[AgentStep]) -> str:
        """Format previous steps for the prompt."""
        if not steps:
            return ""

        formatted = []
        for step in steps:
            formatted.append(f"""
Step {step.step_number}:
  Thought: {step.thought}
  Action: {step.action.value}
  Tool: {step.tool_used or 'N/A'}
  Observation: {step.observation[:300]}{'...' if len(step.observation) > 300 else ''}
""")

        return "\n".join(formatted)

    def get_trace_log(self) -> str:
        """Get formatted trace log for the current execution."""
        if not self.current_trace:
            return "No execution trace available"

        return json.dumps(self.current_trace.to_dict(), indent=2)

    def chat(self, message: str, code: Optional[str] = None) -> str:
        """
        Simple chat interface for the agent.

        Args:
            message: User message
            code: Optional code to analyze

        Returns:
            Agent's response
        """
        response, _ = self.run(message, code)
        return response

    def ingest_knowledge(self, documents_path: str) -> int:
        """
        Ingest documents into the knowledge base.

        Args:
            documents_path: Path to documents directory

        Returns:
            Number of chunks ingested
        """
        from pathlib import Path
        return self.retrieval.ingest_from_directory(Path(documents_path))
