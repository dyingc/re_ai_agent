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
    Plan,
    ToolCallComprehensive,
    ToolCallComprehensiveHistory,
    Critique,
    CritiqueHistory,
)

import utils.utils
from utils .utils import (
    extract_schema,
    get_config,
    MissionAccomplishedToolInput,
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
    insights: List[str] = Field(
        default_factory=list,
        description="Insights or observations derived from previous tool call results. They should be relevant to the task and be updated by comprehend node."
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
    mission_accomplished: Optional[MissionAccomplishedToolInput] = Field(
        default=None,
        description="Final answer to the mission."
    )

    def __hash__(self):
        # Explicitly define __hash__ to handle nested frozen models
        return hash((
            self.task,
            tuple(self.reflections.history),  # Convert lists to tuples for hashability
            tuple(self.analyses.history)
        ))

class REAgent():
    def __init__(self, task: str, analyzing_tools: List[StructuredTool],
                 model_name: str, api_key: str,
                 config: Dict[str, Any],
                 temperature:float=0.0,
                 max_calls: int=15,
                 exit_tool: List[StructuredTool]=[],
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

        self.analyzing_tools = analyzing_tools
        self.exit_tool = exit_tool
        self.analyzing_tool_descs = '\n'.join([str({"name": t.name, "descripition": t.description,
                                          "arguments": t.args_schema.model_fields})
                                     for t in self.analyzing_tools])
        self.tool_names = [t.name for t in self.analyzing_tools] if self.analyzing_tools else []
        self.max_calls = max_calls
        self.num_of_calls = 0
        self.python_tool_name = config.get('agent_config').get('python_tool_name')
        self.exit_tool_name = config.get('agent_config').get('exit_tool_name')

    def analyze(self, state: AgentState) -> AgentState:
        system_msg, task_msgs = self._prepare_analysis_prompts(state)
        analyzer = self._configure_analyzer_llm()
        
        analysis_result = self._perform_analysis_loop(
            system_msg, task_msgs, analyzer, state
        )
        
        if analysis_result.get("mission_accomplished"):
            return analysis_result
            
        return self._handle_successful_analysis(analysis_result, state)

    def _prepare_analysis_prompts(self, state: AgentState) -> tuple[SystemMessage, list[HumanMessage]]:
        config = get_config()
        system_str = config.get('messages').get('analyst').get('system')
        system = SystemMessage(content=system_str)
        
        if state.analysis_needs_improvements and state.critiques.get_latest_critique():
            return system, self._prepare_critique_based_prompt(state, config)
        return system, self._prepare_standard_prompt(state, config)

    def _prepare_standard_prompt(self, state: AgentState, config: dict) -> list[HumanMessage]:
        context = config.get('messages').get('analyst').get('context').format(
            plan="\n".join([f"- {step}" for step in state.plan.plan]),
            plan_step=state.plan.next_step_task,
            insights=state.insights,
            pitfalls="\n".join([f"- {p}" for p in state.plan.common_pitfalls])
        )
        tool_ref = config.get('messages').get('analyst').get('tool_call_reference').format(
            previous_tool_calls=state.tool_call_history.get_relevant_tool_call_n_results_repr(
                state.plan.relevant_tool_results or []
            )
        )
        return [HumanMessage(content=state.task + "\n\n" + context + "\n\n" + tool_ref)]

    def _prepare_critique_based_prompt(self, state: AgentState, config: dict) -> list[HumanMessage]:
        critique = state.critiques.get_latest_critique()
        context = config.get('messages').get('analyst').get('latest_reflection').format(
            latest_tool_call_repr=state.analyses.get_latest_analysis().get_tool_call_expr(),
            chosen_tool_call=critique.chosen_tool,
            detailed_instructions=critique.detailed_instructions,
            relevant_tool_calls_n_results=state.tool_call_history.get_relevant_tool_call_n_results_repr(
                critique.relevant_tool_call_indices
            )
        )
        return [HumanMessage(content=state.task + "\n\n" + context)]

    def _configure_analyzer_llm(self):
        analyzer = self.llm.model_copy()
        analyzer.name = "Analyzer"
        return analyzer.bind_tools(self.analyzing_tools + self.exit_tool)

    def _perform_analysis_loop(self, system_msg: SystemMessage, 
                             task_msgs: list[HumanMessage],
                             analyzer: ChatDeepSeek,
                             state: AgentState) -> dict:
        MAX_ATTEMPTS = 3
        for _ in range(MAX_ATTEMPTS):
            response = analyzer.invoke([system_msg] + task_msgs)
            tool_call = response.tool_calls[0] if response.tool_calls else None
            
            if self._handle_exit_tool(tool_call):
                return {"mission_accomplished": MissionAccomplishedToolInput(**tool_call.get('args'))}
                
            validation_result = self._validate_tool_call(tool_call, state, task_msgs, response)
            if validation_result == "valid":
                return {"response": response, "tool_call": tool_call}
            if validation_result == "retry":
                continue
                
        raise ValueError(f"Failed to analyze after {MAX_ATTEMPTS} attempts")

    def _validate_tool_call(self, tool_call: Optional[dict], 
                          state: AgentState,
                          task_msgs: list, 
                          response: AIMessage) -> str:
        if not tool_call:
            self._handle_missing_tool_call(task_msgs, response)
            return "invalid"
            
        if tool_call.get('name') == self.python_tool_name:
            return self._validate_python_tool_call(tool_call, task_msgs, response)
            
        if state.analyses.duplicate_tool_call(tool_call):
            self._handle_duplicate_tool_call(tool_call, task_msgs, response)
            return "retry"
            
        return "valid"

    def _handle_exit_tool(self, tool_call: Optional[dict]) -> bool:
        if tool_call and tool_call.get('name') == self.exit_tool_name:
            if tool_call.get('args').get('final_answer') and tool_call.get('args').get('evidence'):
                return True
        return False

    def _validate_python_tool_call(self, tool_call: dict, 
                                 task_msgs: list, 
                                 response: AIMessage) -> str:
        try:
            compile(tool_call.get('args').get('code'), '<string>', 'exec')
            return "valid"
        except Exception as e:
            task_msgs.extend([
                response,
                HumanMessage(content=f"Python语法错误: {e}")
            ])
            return "retry"

    def _handle_missing_tool_call(self, task_msgs: list, response: AIMessage):
        task_msgs.extend([
            response,
            HumanMessage(content="请调用一个可用工具来继续分析")
        ])

    def _handle_duplicate_tool_call(self, tool_call: dict,
                                  task_msgs: list,
                                  response: AIMessage):
        error_msg = "该工具调用已被执行过，请分析现有结果并尝试不同的方法"
        task_msgs.extend([
            response,
            ToolMessage(content=error_msg, name=tool_call.get('name'), tool_call_id=tool_call.get('id')),
            HumanMessage(content=error_msg)
        ])

    def _handle_successful_analysis(self, result: dict, state: AgentState) -> dict:
        analysis = Analysis(
            analysis=result["response"].content,
            tool_call=result["tool_call"]
        )
        state.analyses.add_analysis(analysis)
        return {"analyses": state.analyses}

    def reflect(self, state: AgentState) -> AgentState:
        config = get_config() # Reload config to ensure we have the latest settings
        system = SystemMessage(content=config.get('messages').get('reflecter').get('system'))
        latest_analysis = state.analyses.get_latest_analysis()
        if not latest_analysis:
            raise ValueError(f"No analysis found to reflect on. Current analyses: {state.analyses.history}")
        proposed_tool_call = latest_analysis.get_tool_call_expr()
        latest_tool_call_n_result_repr = state.tool_call_history.get_latest_tool_call_result().get_tool_call_n_result_repr() if state.tool_call_history.get_latest_tool_call_result() else ""
        previous_tool_calls_indices = list(range(1, len(state.tool_call_history.history))) if len(state.tool_call_history.history) > 1 else []
        task_str = config.get('messages').get('reflecter').get('task').format(
            problem=state.task,
            insights=state.insights,
            analyzing_tools=self.analyzing_tool_descs,
            analysis=latest_analysis.analysis,
            proposed_tool_call=proposed_tool_call,
            latest_tool_call_result_repr=latest_tool_call_n_result_repr,
            previous_tool_calls=state.tool_call_history.get_relevant_tool_call_repr(previous_tool_calls_indices)
        )
        reflecter = self.llm
        reflecter.name = "Reflecter"
        reflecter = reflecter.bind_tools([ReflectionResult])
        response: AIMessage = reflecter.invoke([system, HumanMessage(content=task_str)])
        reflection = ReflectionResult(**response.tool_calls[0].get('args'))
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
        result = self.analyzing_tools[self.tool_names.index(tool_name)].invoke(input=tool_args)
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
        config = get_config() # Reload config to ensure we have the latest settings
        last_tool_call_result.refine_tool_result(llm=self.llm, config=config)
        return {"tool_call_history": tool_call_history}
        
    def create_plan(self, state: AgentState) -> AgentState:
        """Create a plan based on the current state of the agent especially the new tool call results"""
        config = get_config() # Reload config to ensure we have the latest settings
        task = state.task
        system = SystemMessage(content=config.get('messages').get('planner').get('system'))
        previous_too_call_insights = state.tool_call_history.get_toolcall_history_insights_repr()
        # Consider only reflections that reject the previous analysis
        rejected_reflections = [r for r in state.reflections.history if not r.is_analysis_accepted]
        str_reflections = "\n".join(
            [f"\n{reflection.get_reflect_repr()}" for reflection in rejected_reflections]
        ) if rejected_reflections else ""

        human_message_str = config.get('messages').get('planner').get('task').format(
            problem=task,
            analyzing_tools=self.analyzing_tool_descs,
            insights=state.insights,
            previous_too_call_insights=previous_too_call_insights,
            reflection_history=str_reflections
        )
        messages = [system, HumanMessage(content=human_message_str)]
        planner = self.llm.model_copy()
        planner.name = "Planner"
        planner = planner.bind_tools([Plan])
        response: AIMessage = planner.invoke(messages)
        plan: Plan = Plan(**response.tool_calls[0].get('args'))
        return {"plan": plan,}

    def comprehend_tool_result(self, state: AgentState) -> AgentState:
        config = get_config() # Reload config to ensure we have the latest settings
        latest_tool_call = state.tool_call_history.history[0] if state.tool_call_history.history else None
        if not latest_tool_call:
            return state # Nothing changed
        existing_insights = state.insights
        num_historical_tool_call = len(state.tool_call_history.history)
        if num_historical_tool_call > 1:
            previous_tool_call_results_repr = state.tool_call_history.get_relevant_tool_call_n_insight_repr(list(range(1, num_historical_tool_call)))
        else:
            previous_tool_call_results_repr = ""
        system = SystemMessage(content=config.get('messages').get('tool_result_miner').get('system').format(task=state.task))
        task_str = config.get('messages').get('tool_result_miner').get('task').format(
            latest_tool_call= latest_tool_call.tool_call_repr,
            latest_tool_call_result=latest_tool_call.get_tool_call_result(),
            existing_insights="\n".join([f"- {insight}" for insight in existing_insights]),
            previous_tool_calls=previous_tool_call_results_repr
        )
        messages = [system, HumanMessage(content=task_str)]
        miner = self.llm.model_copy()
        miner.name = "ToolResultMiner"
        miner = miner.bind_tools([ToolCallComprehensive])
        response: AIMessage = miner.invoke(messages)
        # Extract ToolCallComprehensive from the response
        comprehensive: ToolCallComprehensive = ToolCallComprehensive(**response.tool_calls[0].get('args'))
        result = dict()
        if comprehensive:
            if comprehensive.latest_finding:
                latest_tool_call.insight = comprehensive.latest_finding
            result['comprehensives'] = state.comprehensives
            result['comprehensives'].add_tool_call_comprehensive(comprehensive)
            if comprehensive.updated_overall_insights:
                result['insights'] = comprehensive.updated_overall_insights
            else:
                result['insights'] = existing_insights

        return result
            
    def toolcall_needs_refinement(self, state: AgentState) -> bool:
        last_tool_call_result = state.tool_call_history.get_latest_tool_call_result()
        # Check if the tool call needs refinement
        refinable_tools = {item.name:item.value for item in AvailableTool}
        refinable_tools = {t: f for t, f in refinable_tools.items() 
                           if t in config.get('agent_config').get('refinable_tools')}
        if last_tool_call_result.orig_tool_call.get('name', '') in refinable_tools.values():
            return True
        return False

    def criticize(self, state: AgentState) -> AgentState:
        config = get_config() # Reload config to ensure we have the latest settings
        _problem = state.task
        _analyzing_tools = self.analyzing_tool_descs
        _latest_tool_call_repr = state.analyses.get_latest_analysis().get_tool_call_expr() if state.analyses.get_latest_analysis() else ""
        _latest_reflection = state.reflections.get_latest_reflection()
        _previous_tool_calls_n_insights = state.tool_call_history.get_toolcall_history_insights_repr()
        system = SystemMessage(content=config.get('messages').get('critic').get('system'))
        task_str = config.get('messages').get('critic').get('task').format(
            problem=_problem,
            insights=state.insights,
            analyzing_tools=_analyzing_tools,
            latest_tool_call_repr=_latest_tool_call_repr,
            latest_reflection=_latest_reflection.get_reflect_repr() if _latest_reflection else "",
            previous_tool_calls_n_insights= _previous_tool_calls_n_insights
        )
        critic = self.llm.model_copy()
        critic.name = "Critic"
        critic = critic.bind_tools([Critique])
        response: AIMessage = critic.invoke([system, HumanMessage(content=task_str)])
        critique: Critique = Critique(**response.tool_calls[0].get('args'))
        critiques = state.critiques.model_copy()
        critiques.add_critique(critique)
        return {"critiques": critiques}

    def analyze_done(self, state: AgentState) -> bool:
        return True if (state.mission_accomplished and state.mission_accomplished.final_answer \
            and state.mission_accomplished.evidence) else False

config = get_config()

# Define analyzing tools from config
analyzing_tools: List[StructuredTool] = [
    getattr(tools.reverse_engineering, tool_name)
    for tool_name in config.get("analyzing_tools", [])
]

exit_tool: List[StructuredTool] = [getattr(utils.utils, tool_name) for tool_name in config.get("existing_tools", [])]

# Create a single instance of the agent at module level
task = config.get('messages').get('analyst').get('task')
security_researcher_agent = REAgent(task=task, analyzing_tools=analyzing_tools,
                                    api_key=api_key,
                                    config=config,
                                    model_name=config.get('agent_config').get('model_name'),
                                    temperature=config.get('agent_config').get('temperature'),
                                    max_calls=config.get('agent_config').get('max_calls'),
                                    exit_tool=exit_tool)

re_graph = security_researcher_agent.graph


def main():
    # Initiate LangSmith
    from langsmith import Client
    os.environ["LANGSMITH_PROJECT"] = "re-agent_from_main"

    _ = Client()

    # Define the task message
    task_message = config["messages"]["analyst"]["task"]
    # Use an automatically generated UUID for the user_id
    graph_config = {"configurable": {"thread_id": str(uuid.uuid4())},
                    "recursion_limit": 50}

    # Since we're using MemorySaver, we can simplify this
    result = re_graph.invoke(input={"task": task_message}, config=graph_config)

if __name__ == "__main__":
    main()
