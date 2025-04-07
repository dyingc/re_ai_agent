# Standard library imports
import operator
import os
import uuid
import yaml, json, ast
from typing import Annotated, Any, Dict, List, Optional, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

## Local imports
from utils.datatypes import (
    ReflectionResult,
    ReflectionHistory,
    Analysis,
    AnalysesHistory,
    ToolCallResult,
    ToolCallResultHistory,
    AvailableTool,
    Insight,
    Plan,
    ToolCallComprehensive,
    ToolCallComprehensiveHistory,
    Critique,
    CritiqueHistory,
)

from utils .utils import (
    extract_schema,
)
    
import tools.reverse_engineering
# import tools.utils
# from tools.utils import AnalysisReflectionResult, reflect_analysis

# Third-party imports
from dotenv import load_dotenv
from langchain_core.messages import (
    ToolCall,
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools.structured import StructuredTool
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

# Load environment variables
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

class AgentState(BaseModel):
    model_config = ConfigDict(frozen=True) # Makes the model hashable

    task: str = Field(..., description="Task to be performed by the agent")
    plan: Plan = Field(
        default_factory=Plan,
        description="Current plan of action for the agent. It should be updated as the agent progresses through its task."
    )
    insights: List[Insight] = Field(
        default_factory=list,
        description="List of insights or observations derived from previous tool call results. They should be relevant to the task and be updated by comprehend node."
    )
    comprehensives : Optional[ToolCallComprehensiveHistory] = Field(
        default_factory=ToolCallComprehensiveHistory,
        description="History of comprehensive tool call results, including insights derived from them."
    )
    reflections: ReflectionHistory = Field(default_factory=ReflectionHistory,
                description="List of all reflections against the agent's analyses.")
    critiques: CritiqueHistory = Field(
        default_factory=CritiqueHistory,
        description="History of critiques made by the agent against its own analyses and reflections.")
    analyses: AnalysesHistory = Field(default_factory=AnalysesHistory,
                description="List of all analyses done by the agent.")
    tool_call_history: ToolCallResultHistory = Field(
        default_factory=ToolCallResultHistory,
        description="History of all tool calls made by the agent."
    )
    analysis_needs_improvements: bool = Field(
        default=False,
        description="Flag to indicate if the latest analysis needs improvements. This is set to True if the latest reflection rejects the analysis."
    )

    def __hash__(self):
        # Explicitly define __hash__ to handle nested frozen models
        return hash((
            self.task,
            tuple(self.reflections.history),  # Convert lists to tuples for hashability
            tuple(self.analyses.history)
        ))

class REAgent():
    def __init__(self, task: str, tools: List[StructuredTool],
                 model_name: str, api_key: str,
                 config: Dict[str, Any],
                 temperature:float=0.0,
                 max_calls: int=15,
                ):
        self.task = task
        self.api_key = api_key
        self.model_name = model_name
        graph = StateGraph(AgentState, input=AgentState(task=task))

        # Initialize the graph with the initial state
        graph.add_node("analyze", self.analyze)
        graph.add_node("reflect", self.reflect)
        graph.add_node("action", self.take_action)
        graph.add_node("tool_call_refine", self.refine_tool_call)
        graph.add_node("criticize", self.criticize)
        graph.add_node("planning", self.create_plan)
        graph.add_node("comprehend", self.comprehend_tool_result)

        # Flow of the agent
        graph.add_edge("criticize", "analyze")
        graph.add_edge("tool_call_refine", "comprehend")
        graph.add_edge("planning", "analyze")
        graph.add_edge("comprehend", "planning")
        graph.add_conditional_edges("reflect", self.reflect_good_to_continue,
                                    {True: "action", False: "criticize"})
        graph.add_conditional_edges("analyze", self.analyze_done,
                                    {True: END, False: "reflect"})
        graph.add_conditional_edges("action", self.toolcall_needs_refinement,
                                    {True: "tool_call_refine", False: "comprehend"})
        graph.set_entry_point("planning")

        # Compile the graph
        self.graph = graph.compile(
            name = "Reverse Engineering Agent",
            debug = True
        )

        # LLM
        self.llm = ChatDeepSeek(model_name=model_name, api_key=api_key, temperature=temperature)

        self.tools = tools
        self.analyzing_tool_descs = '\n'.join([str({"name": t.name, "descripition": t.description,
                                          "arguments": t.args_schema.model_fields})
                                     for t in self.tools])
        self.tool_names = [t.name for t in self.tools] if self.tools else []
        self.max_calls = max_calls
        self.num_of_calls = 0
        self.python_tool_name = config.get('agent_config').get('python_tool_name')

    def analyze(self, state: AgentState) -> AgentState:
        system_str = config.get('messages').get('analyst').get('system')
        task_str = state.task
        relevant_tool_call_indices = state.plan.relevant_tool_results if state.plan.relevant_tool_results else []
        previous_tool_calls_repr = state.tool_call_history.get_relevant_tool_call_n_results_repr(relevant_tool_call_indices)
        plan_step = state.plan.next_step_task
        insights_repr = "\n".join([f"- {insight}" for insight in state.insights])
        common_pitfalls_repr = "\n".join([f"- {pitfall}" for pitfall in state.plan.common_pitfalls])
        critique = state.critiques.get_latest_critique()
        if not state.analysis_needs_improvements or not critique:
            context = "\n\n" + config.get('messages').get('analyst').get('context').format(
                plan_step=plan_step,
                insights=insights_repr,
                pitfalls=common_pitfalls_repr)
            tool_call_reference = "\n\n" + config.get('messages').get('analyst').get('tool_call_reference').format(
                previous_tool_calls=previous_tool_calls_repr
            )
            task = [HumanMessage(content=task_str + context + tool_call_reference)]
        else:
            latest_tool_call_repr = state.analyses.get_latest_analysis().get_tool_call_expr() if state.analyses.get_latest_analysis() else ""
            chosen_tool_call = state.critiques.get_latest_critique().chosen_tool
            detailed_instructions = state.critiques.get_latest_critique().detailed_instructions
            relevant_tool_call_indices = state.critiques.get_latest_critique().relevant_tool_call_indices
            previous_tool_calls_repr = state.tool_call_history.get_relevant_tool_call_n_results_repr(relevant_tool_call_indices)
            context = "\n\n" + config.get('messages').get('analyst').get('latest_reflection').format(
                latest_tool_call_repr=latest_tool_call_repr,
                chosen_tool_call=chosen_tool_call,
                detailed_instructions=detailed_instructions,
                relevant_tool_calls_n_results=previous_tool_calls_repr
            )
            task = [HumanMessage(content=task_str + context)]
        system = SystemMessage(content=system_str)
        MAX = 3
        SUC = False
        analyzer = self.llm.model_copy()
        analyzer.name = "Analyzer"
        analyzer = analyzer.bind_tools(self.tools)
        for _ in range(MAX): # Try 3 times to ensure code syntax (if Python tool is used)
            if SUC:
                break
            response: AIMessage = analyzer.invoke([system] + task)
            analysis: Analysis = extract_schema(Analysis, llm=self.llm, ai_response=response, config=config)
            tool_call = analysis.tool_call
            # Ensure the syntax is correct if it's calling the Python interpreter
            if tool_call.get('name', '') == self.python_tool_name:
                code_str = tool_call.get('args', {}).get('code', '')
                try:
                    compile(code_str, '<string>', 'exec')
                    SUC = True
                except Exception as e:
                    task.append(response) # If there's a syntax error, ask the LLM to fix it
                    task.append(HumanMessage(content=f"Error: {e}"))
            else:
                SUC = True
        if not SUC:
            raise ValueError(f"Failed to analyze the code after {MAX} attempts.")
        state.analyses.add_analysis(analysis)
        return {
            "analyses": state.analyses
        }

    def reflect(self, state: AgentState) -> AgentState:
        system = SystemMessage(content=config.get('messages').get('reflecter').get('system'))
        latest_analysis = state.analyses.get_latest_analysis()
        if not latest_analysis:
            raise ValueError(f"No analysis found to reflect on. Current analyses: {state.analyses.history}")
        proposed_tool_call = latest_analysis.get_tool_call_expr()
        latest_tool_call_n_result_repr = state.tool_call_history.get_latest_tool_call_result().get_tool_call_n_result_repr() if state.tool_call_history.get_latest_tool_call_result() else ""
        previous_tool_calls_indices = list(range(1, len(state.tool_call_history.history))) if len(state.tool_call_history.history) > 1 else []
        task_str = config.get('messages').get('reflecter').get('task').format(
            problem=state.task,
            analyzing_tools=self.analyzing_tool_descs,
            proposed_tool_call=proposed_tool_call,
            tool_call_reasoning=latest_analysis.reason_for_tool_call,
            latest_tool_call_result_repr=latest_tool_call_n_result_repr,
            previous_tool_calls=state.tool_call_history.get_relevant_tool_call_repr(previous_tool_calls_indices)
        )
        reflecter = self.llm
        reflecter.name = "Reflecter"
        response: AIMessage = reflecter.invoke([system, HumanMessage(content=task_str)])
        reflection: ReflectionResult = extract_schema(ReflectionResult,
                                                      llm=reflecter,
                                                      ai_response=response,
                                                      config=config)
        state.reflections.add_reflection(reflection)
        return {"reflections": state.reflections, "analysis_needs_improvements": not reflection.is_analysis_accepted}

    def reflect_good_to_continue(self, state: AgentState) -> bool:
        reflection = state.reflections.get_latest_reflection()
        if not reflection:
            return True # Continue by default
        return reflection.is_analysis_accepted

    def _execute_tool(self, tool_call: ToolCall) -> ToolMessage:
        """Execute a single tool call.

        Args:
            tool_call: The tool call to execute

        Returns:
            ToolMessage containing the execution result
        """
        tool_name = tool_call.get('name', '')
        tool_args = tool_call.get('args', {})
        if not tool_name:
            raise ValueError(f"Tool name is missing in the tool call: {str(tool_call)}")
        result = self.tools[self.tool_names.index(tool_name)].invoke(input=tool_args)
        return ToolMessage(
            content=result,
            name=tool_name,
            tool_call_id=tool_call.get('tool_call_id', ''),
        )

    def take_action(self, state: AgentState) -> AgentState:
        tool_call = state.analyses.get_latest_analysis().tool_call

        # Only process the first valid tool call
        if tool_call:
            try:
                tool_call_response = self._execute_tool(tool_call)
            except Exception as e:
                error_message = f"Tool execution failed: {str(e)}"
                tool_call_response: ToolMessage = ToolMessage(
                        content=error_message,
                        name=tool_call.get('name', ''),
                        tool_call_id=tool_call.get('tool_call_id', ''),
                )
        else:
            tool_call_response: ToolMessage = ToolMessage(
                content="No tool calls found",
                name="",
                tool_call_id="",
            )
        tool_call_result = ToolCallResult(
            orig_tool_call=tool_call,
            raw_tool_result=tool_call_response,
        )
        tool_call_history = state.tool_call_history.model_copy()
        tool_call_history.add_tool_call_result(tool_call_result)
        return {"tool_call_history": tool_call_history}

    def refine_tool_call(self, state: AgentState) -> AgentState:
        tool_call_history = state.tool_call_history.model_copy()
        last_tool_call_result = tool_call_history.get_latest_tool_call_result()
        last_tool_call_result.refine_tool_result()
        return {"tool_call_history": tool_call_history}
        
    def create_plan(self, state: AgentState) -> AgentState:
        """Create a plan based on the current state of the agent especially the new tool call results"""
        task = state.task
        system = SystemMessage(content=config.get('messages').get('planner').get('system'))
        str_insights = "\n".join(["- " + insight.get_insight_string() for insight in state.insights])
        previous_too_call_results = state.tool_call_history.get_history_repr()
        # Consider only reflections that reject the previous analysis
        rejected_reflections = [r for r in state.reflections.history if not r.is_analysis_accepted]
        str_reflections = "\n".join(
            [f"- {reflection.get_reflection_string()}" for reflection in rejected_reflections]
        ) if rejected_reflections else ""

        human_message_str = config.get('messages').get('planner').get('task').format(
            problem=task,
            analyzing_tools=self.analyzing_tool_descs,
            insights=str_insights,
            previous_tool_call_results=previous_too_call_results,
            reflection_history=str_reflections
        )
        messages = [system, HumanMessage(content=human_message_str)]
        planner = self.llm.model_copy()
        planner.name = "Planner"
        response: AIMessage = planner.invoke(messages)
        plan: Plan = extract_schema(Plan, llm=planner, ai_response=response, config=config)
        return {"plan": plan,}

    def comprehend_tool_result(self, state: AgentState) -> AgentState:
        latest_tool_call = state.tool_call_history.history[0] if state.tool_call_history.history else None
        if not latest_tool_call:
            return state # Nothing changed
        existing_insights = [insight.get_insight_string() for insight in state.insights]
        existing_insights = "\n".join([f"- {insight}" for insight in existing_insights if insight.strip()])
        num_historical_tool_call = len(state.tool_call_history.history)
        if num_historical_tool_call > 1:
            previous_tool_call_results_repr = state.tool_call_history.get_relevant_tool_call_n_results_repr(list(range(1, num_historical_tool_call)))
        else:
            previous_tool_call_results_repr = ""
        system = SystemMessage(content=config.get('messages').get('tool_result_miner').get('system').format(task=state.task))
        task_str = config.get('messages').get('tool_result_miner').get('task').format(
            latest_tool_call= latest_tool_call.tool_call_repr,
            latest_tool_call_result=latest_tool_call.get_tool_call_result(),
            existing_insights=existing_insights,
            previous_tool_calls=previous_tool_call_results_repr
        )
        messages = [system, HumanMessage(content=task_str)]
        miner = self.llm.model_copy()
        miner.name = "ToolResultMiner"
        response: AIMessage = miner.invoke(messages)
        # Extract ToolCallComprehensive from the response
        comprehensive: ToolCallComprehensive = extract_schema(
            ToolCallComprehensive,
            llm=miner,
            ai_response=response,
            config=config
        )
        result = dict()
        if comprehensive:
            result['comprehensive'] = state.comprehensives
            result['comprehensive'].add_tool_call_comprehensive(comprehensive)
            if comprehensive.need_to_update_insights:
                # Update insights only if the comprehensive indicates it
                result['insights'] = comprehensive.updated_insights

        return result
            

    def toolcall_needs_refinement(self, state: AgentState) -> bool:
        last_tool_call_result = state.tool_call_history.get_latest_tool_call_result()
        # Check if the tool call needs refinement
        refinable_tools = {item.name:item.value for item in AvailableTool}
        refinable_tools = [t for t in refinable_tools 
                           if t in config.get('agent_config').get('refinable_tools')]
        if last_tool_call_result.orig_tool_call.get('name', '') in refinable_tools:
            return True
        return False

    def criticize(self, state: AgentState) -> AgentState:
        _problem = state.task
        _analyzing_tools = self.analyzing_tool_descs
        _latest_tool_call_repr = state.analyses.get_latest_analysis().get_tool_call_expr() if state.analyses.get_latest_analysis() else ""
        _latest_reflection = state.reflections.get_latest_reflection()
        _previous_tool_calls_n_results = state.tool_call_history.get_history_repr()
        system = SystemMessage(content=config.get('messages').get('critic').get('system'))
        task_str = config.get('messages').get('critic').get('task').format(
            problem=_problem,
            analyzing_tools=_analyzing_tools,
            latest_tool_call_repr=_latest_tool_call_repr,
            latest_reflection=_latest_reflection.get_reflect_repr() if _latest_reflection else "",
            previous_tool_calls_n_results= _previous_tool_calls_n_results
        )
        critic = self.llm.model_copy()
        critic.name = "Critic"
        response: AIMessage = critic.invoke([system, HumanMessage(content=task_str)])
        critique: Critique = extract_schema(Critique, llm=critic, ai_response=response, config=config)
        critiques = state.critiques.model_copy()
        critiques.add_critique(critique)
        return {"critiques": critiques}
            

    def analyze_done(self, state: AgentState) -> bool:
        analysis = state.analyses.get_latest_analysis()
        return analysis.is_task_resolved

# Load configuration from YAML file
with open("config.yaml") as f:
    config = yaml.safe_load(f)


# Define analyzing tools from config
analyzing_tools: List[StructuredTool] = [
    getattr(tools.reverse_engineering, tool_name)
    for tool_name in config.get("analyzing_tools", [])
]

# Create a single instance of the agent at module level
task = config.get('messages').get('analyst').get('task')
security_researcher_agent = REAgent(task=task, tools=analyzing_tools,
                                    api_key=api_key,
                                    config=config,
                                    model_name=config.get('agent_config').get('model_name'),
                                    temperature=config.get('agent_config').get('temperature'),
                                    max_calls=config.get('agent_config').get('max_calls'))

re_graph = security_researcher_agent.graph


def main():
    # Initiate LangSmith
    from langsmith import Client
    os.environ["LANGSMITH_PROJECT"] = "re-agent_from_main"

    _ = Client()

    # Define the task message
    task_message = config["messages"]["analyst"]["task"]
    # Use an automatically generated UUID for the user_id
    user_id = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Since we're using MemorySaver, we can simplify this
    result = re_graph.invoke(input={"task": task_message}, config=user_id)

if __name__ == "__main__":
    main()

