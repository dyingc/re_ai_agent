from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    Field,
    field_validator,
    model_validator
)
from typing import List, Optional, ClassVar, TypedDict, Annotated, Dict
from enum import Enum
import yaml, ast
import sys
from pathlib import Path

# Add the project root to PYTHONPATH if not already present
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
import tools.reverse_engineering

import tools.reverse_engineering
from langchain_core.tools.structured import StructuredTool
from langchain_core.runnables import Runnable
from langchain_core.messages import (
    ToolCall,
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# Load configuration from YAML file
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Define the available tools
def _create_tool_enum(tool_names: List[str]):
    return Enum("AvailableTool", 
                {name: getattr(tools.reverse_engineering, name).name for name in tool_names},
                type=str)

def _get_tool_call_repr(tool_call: ToolCall) -> str:
    args = tool_call.get('args', {})
    args_str = ", ".join([
        f'{k} = "{v}"' if isinstance(v, str) else f'{k} = {str(v)}'
        for k, v in args.items()])
    code_str = f"{tool_call['name']} ({args_str})"
    return code_str

AvailableTool = _create_tool_enum(config["analyzing_tools"])

class ReflectionResult(BaseModel):
    model_config = ConfigDict(frozen=True) # Makes the model hashable

    is_analysis_accepted: bool = Field(
        default=True,
        description="Shall we accepted the analysis provided by the junior analyzer?")
    issue_in_the_analysis: str = Field("The issue or places need to improve in the previous analysis submitted by the analyzer.")
    recommended_tools: List[AvailableTool] = Field( # type: ignore
        description=f"The names of the recommended tools based on the current analysis result and available tool list. "\
            "Those tools are most like to be the candidate of the next tool call by the analyzer."\
            " At most two tools shall be recommended.")
    next_step_task: str = Field("Suggested follow-up actions or tool alternatives")

    def get_reflect_repr(self)->str:
        return f"Reflection: {self.issue_in_the_analysis}\n" + \
               f"High Quality to Continue: {self.is_analysis_accepted}\n" + \
               f"Recommended Tools: {', '.join([tool.name for tool in self.recommended_tools])}\n" + \
               f"Next Step Task: {self.next_step_task}"

    @field_validator('recommended_tools')
    def check_recommended_tools(cls, v):
        if len(v) > 2:
            raise ValueError('At most two tools shall be recommended.')
        return v

class ReflectionHistory(BaseModel):
    model_config = ConfigDict(frozen=True) # Makes the model hashable

    history: List[ReflectionResult] = Field(default_factory=list,
        description="List of reflections against the agent's analysis. Latest reflection first.")
    def add_reflection(self, reflection: ReflectionResult):
        self.history.insert(0, reflection)
    def get_latest_reflection(self) -> ReflectionResult:
        return self.history[0] if self.history else None
    def get_history_repr(self) -> str:
        """Get a string representation of the history of reflections."""
        return "\n".join([f"Reflection {i}: \n'''\n{self.history[i].get_reflect_repr()}\n'''\n" 
                          for i in range(len(self.history))])
    def get_latest_decision(self) -> bool:
        return self.history[0].is_analysis_accepted if self.history else True

class Analysis(BaseModel):
    MAX_LEN_OF_ANALYSIS: ClassVar[int] = 1000
    _tool_call: Optional[ToolCall] = PrivateAttr(default=None) # Proposed tool call.
    reason_for_tool_call: Optional[str] = Field(default=None,
        description="The reason or explanation of the purpose of the tool call. Leave it empty if no reason provided.")
    is_task_resolved: bool = Field(default=False,
        description="If the task is confirmed to be completely resolved. Though proposing a tool call is generally regarded as 'not resolved', no tool call isn't necessarily mean the task is resolved, unless this can be derived from the response.")
    final_answer: Optional[str] = Field(
        default=None,
        description="The final answer or conclusion derived from the analysis. This should only be set if the task is confirmed to be resolved."
    )

    def set_tool_call(self, tool_call: ToolCall):
        self._tool_call = tool_call
    def get_tool_call(self) -> Optional[ToolCall]:
        return self._tool_call
    def get_tool_call_expr(self) -> Optional[str]:
        if not self._tool_call:
            return None
        return _get_tool_call_repr(self._tool_call)

    @model_validator(mode="after")
    def check_tool_call(self):
        # Check the final answer if the task is resolved
        if self.is_task_resolved and not self.final_answer:
            raise ValueError('If the task is resolved, a final answer must be provided.')
        if self.final_answer and len(self.final_answer.strip()) == 0 and self.is_task_resolved:
            raise ValueError('Final answer cannot be an empty string if the task is resolved.')
        if not self.is_task_resolved and (self.final_answer is not None and self.final_answer.strip() != ""):
            raise ValueError('Final answer should not be set if the task is not resolved.')
        return self

class AnalysesHistory(BaseModel):
    model_config = ConfigDict(frozen=True) # Makes the model hashable

    history: List[Analysis] = Field(default_factory=list,
        description="List of analyses against the agent's task. Latest analysis first.")
    def add_analysis(self, analysis: Analysis):
        self.history.insert(0, analysis)
    def get_latest_analysis(self) -> Analysis:
        return self.history[0] if self.history else None
    def get_history(self) -> List[Analysis]:
        return self.history
    def get_repr_of_history_tool_calls(self) -> List[str]:
        return [ana.get_tool_call_expr() for ana in self.history]
    def get_latest_decision(self) -> bool:
        return self.history[0].is_task_resolved if self.history else False

class ImprovedPseudoCode(TypedDict):
    refined_code: Annotated[str, "The improved pseudo code after refinement."]

class ToolCallResult(BaseModel):
    orig_tool_call: ToolCall = Field(description="The original tool call.")
    raw_tool_result: ToolMessage = Field(description="The RAW result of the tool call.")
    tool_result_content: Optional[str]  = Field(default=None,
        description="The string content of the tool result.")
    tool_call_repr: Optional[str] = Field(default=None,
        description="The string representation of the tool call.")
    refined_tool_result: Optional[ImprovedPseudoCode] = Field(default=None,
        description="The refined result of the 'pseudo_code_tool' tool call.")

    def __init__(self, **data):
        super().__init__(**data)
        self.tool_result_content = self.raw_tool_result.content
        self.tool_call_repr = self._get_tool_call_expr()

    def get_tool_call_n_result_repr(self) -> str:
        result = self.get_tool_call_result()
        call = self.tool_call_repr
        return f"Tool Call: \n'''\n{call}\n'''\nTool Call Result:\n'''\n{result}\n'''".strip() if call else result.strip()
    def get_tool_call_result(self) -> str:
        tool_result_content_dict = ast.literal_eval(self.tool_result_content)
        original_result = tool_result_content_dict.get('result', '')
        if self.refined_tool_result:
            total_result = config.get('messages').get('tool_result_refiner').get('pseudo_code_refiner').get('total_pseudo_code_result').format(
                original_pseudo_code=original_result,
                refined_pseudo_code=self.refined_tool_result.get('refined_code')
            )
            return total_result
        return original_result
    def _get_tool_call_expr(self) -> Optional[str]: # No result
        if not self.orig_tool_call:
            return None
        return _get_tool_call_repr(self.orig_tool_call)

    def refine_tool_result(self, llm: Runnable, config: dict = config):
        # Check if the tool call needs refinement
        refinable_tools = {item.name:item.value for item in AvailableTool}
        refinable_tools = {t: f for t, f in refinable_tools.items() 
                           if t in config.get('agent_config').get('refinable_tools')}
        if self.orig_tool_call.get('name', '') in refinable_tools.values():
            SUC = False
            MAX = config.get('agent_config', {}).get('max_tool_result_refinement_attempts', 3)
            while not SUC and MAX > 0:
                # LLM bound with the structured output
                refiner = llm.copy().bind_tools([ImprovedPseudoCode], tool_choice="ImprovedPseudoCode")
                refiner.name = "Pseudo Code Refiner"
                system = SystemMessage(content=config.
                                    get('messages').get('tool_result_refiner', {}).
                                    get('pseudo_code_refiner', {}).get('system', ''))
                task = HumanMessage(content=config.get('messages').get('tool_result_refiner', {}).
                                    get('pseudo_code_refiner', {}).get('task', '').
                                    format(
                                        pseudo_code_output=self.tool_result_content
                                    ))
                response: AIMessage = refiner.invoke([system, task])
                # Get structured output from the response
                self.refined_tool_result = ImprovedPseudoCode(**response.tool_calls[0].get('args', {})) if response.tool_calls else None
                if self.refined_tool_result:
                    SUC = True

class ToolCallResultHistory(BaseModel):
    history: List[ToolCallResult] = Field(default_factory=list,
        description="List of tool call results. Latest tool call result first.")
    def add_tool_call_result(self, tool_call_result: ToolCallResult):
        self.history.insert(0, tool_call_result)
    def get_latest_tool_call_result(self) -> ToolCallResult:
        return self.history[0] if self.history else None
    def get_history_repr(self) -> str:
        """Get a string representation of the history of tool call and results."""
        return self.get_relevant_tool_call_n_results_repr(list(range(len(self.history))))
    def get_relevant_tool_call_n_results_repr(self, indices: List[int]) -> str:
        """Including both the tool call representation and the result for the specified indices."""
        if not indices:
            return ""
        relevant_history = [self.history[i] for i in indices if 0 <= i < len(self.history)]
        return "\n".join([f"Tool Call {i}: \n'''\n{relevant_history[i].tool_call_repr}\n'''\nTool Call {i} Result:\n'''\n{relevant_history[i].get_tool_call_result()}\n'''\n" 
                          for i in range(len(relevant_history)) if relevant_history[i].tool_call_repr])
    def get_relevant_tool_call_repr(self, indices: List[int]) -> str:
        """Get the string representation of the tool calls (without results) for the specified indices."""
        if not indices:
            return ""
        relevant_history = [self.history[i] for i in indices if 0 <= i < len(self.history)]
        return "\n".join([f"Tool Call {i}: \n'''\n{relevant_history[i].tool_call_repr}\n'''\n" 
                          for i in range(len(relevant_history)) if relevant_history[i].tool_call_repr])

# class Insight(BaseModel):
#     insight: str = Field(..., description="The insight or observation derived from the analysis or tool results. It should be closely related to the task and provide meaningful information. It can be a hypothesis, a conclusion, or an observation that helps in understanding the task better or releasing some new logic or idea.")
#     relevance_score: float = Field(..., ge=0.0, le=1.0, description="A score indicating the relevance of this insight to the task at hand. 0-1 scale (higher means more relevant/important to the task). It can be changed while re-evaluating the insight based on new information or critiques.")
#     evidence: Optional[str] = Field(None, description="Optional evidence or reasoning supporting the insight. This can be a reference to tool results or analysis.")

#     def _get_relevance_label(self) -> str:
#         if self.relevance_score < 0.5:
#             return "Low"
#         elif self.relevance_score < 0.75:
#             return "Medium"
#         else:
#             return "High"

#     def get_insight_string(self) -> str:
#         relevance_label = self._get_relevance_label()
#         return f"Insight: {self.insight} (Relevance: {relevance_label})" + (
#             f" | Evidence: {self.evidence}" if self.evidence else ""
#         )

class ToolCallComprehensive(BaseModel):
    latest_finding: str = Field(..., description="The new information derived from the latest tool result.")
    updated_insights: Optional[str] = Field(
        default="",
        description="Insights or observations derived from all previous tool results, including the latest finding."
    )

class ToolCallComprehensiveHistory(BaseModel):
    history: List[ToolCallComprehensive] = Field(default_factory=list,
        description="List of tool-call-comprehensives. Latest comprehensive tool call result first.")

    def add_tool_call_comprehensive(self, tool_call_comprehensive: ToolCallComprehensive):
        self.history.insert(0, tool_call_comprehensive)

    def get_latest_tool_call_comprehensive(self) -> ToolCallComprehensive:
        return self.history[0] if self.history else None

    def get_history(self) -> List[ToolCallComprehensive]:
        return self.history

    def get_latest_updated_insights(self) -> str:
        """Get the latest updated insights, empty string if unavailable."""
        for comprehensive in self.history:
            if len(comprehensive.updated_insights.strip()) > 0:
                return comprehensive.updated_insights
        return ""

class Plan(BaseModel):
    model_config = ConfigDict(frozen=True)  # Makes the model hashable
    plan: List[str] = Field(
        default_factory=list,
        description="A step-by-step analysis plan that clearly guides the junior analyst through the investigation process, outlining what to do and in what order."
    )
    next_step_task: str = Field(
        default="",
        description="A thorough explanation of the immediate next **one** step, including key considerations, any sub-tasks involved, and the reasoning or logic behind the recommended implementation."
    )
    relevant_tool_results: Optional[List[int]] = Field(
        default_factory=list,
        description="If there are results from previous tool calls, identify the ones most relevant to the current step and list their **indices** (no need for the full contents). These should help inform the next tool call. If none are relevant, say so clearly."
    )
    common_pitfalls: Optional[List[str]] = Field(
        default_factory=list,
        description="A summary of common mistakes the junior analyst should avoid, based on previous reflections and known challenges encountered in past analyses. Highlight these to help steer clear of repeated errors."
    )

    @field_validator('next_step_task')
    def check_next_step_task(cls, v):
        if not v.strip():
            raise ValueError('Next step task cannot be empty.')
        return v.strip()
    @field_validator('plan')
    def check_plan(cls, v):
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError('Plan must be a non-empty list of actionable steps.')
        return v
    @field_validator('common_pitfalls')
    def check_common_pitfalls(cls, v):
        if not isinstance(v, list):
            raise ValueError('Common pitfalls must be a list.')
        if len(v) > 3:
            # At most three common pitfalls are allowed
            v = v[:3]  # Limit to three common pitfalls
        return v

class Critique(BaseModel):
    model_config = ConfigDict(frozen=True)  # Makes the model hashable
    chosen_tool: AvailableTool = Field( # type: ignore
        description="The tool chosen by the critic for the analysis step. It should be one of the available tools."
    )
    detailed_instructions: str = Field(...,
        description="Detailed Instructions: Provide thorough, step-by-step guidance for the junior analyst to follow. These instructions should be comprehensive, covering all necessary context, methodologies, tools, and expectations required to perform a high-quality analysis. Be explicit and detailed—even verbose—if needed, to ensure nothing is left to interpretation. Assume the analyst needs clarity on both what to do and why it’s important. Don’t worry about brevity; completeness is the priority."
    )
    relevant_tool_call_indices: Optional[List[int]] = Field(
        default_factory=list,
        description="Indices of the most relevant tool calls that support the critique."
    )

    @field_validator('chosen_tool')
    def check_chosen_tool(cls, v):
        if not isinstance(v, AvailableTool):
            raise ValueError('Chosen tool must be an instance of AvailableTool enum.')
        return v
    
class CritiqueHistory(BaseModel):
    model_config = ConfigDict(frozen=True)  # Makes the model hashable
    history: List[Critique] = Field(default_factory=list,
        description="List of critiques against the agent's analysis. Latest critique first.")

    def add_critique(self, critique: Critique):
        self.history.insert(0, critique)
    def get_latest_critique(self) -> Critique:
        return self.history[0] if self.history else None


def main():
    # Load environment variables
    from dotenv import load_dotenv
    import os, yaml
    # Load configuration from YAML file
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    from langchain_deepseek import ChatDeepSeek
    model_name = config.get('agent_config').get('model_name')
    temperature=config.get('agent_config').get('temperature')
    llm = ChatDeepSeek(model_name=model_name, api_key=api_key, temperature=temperature)
    # Test ToolCallResult
    tool_call = ToolCall(name="get_function_list", args={"binary_path": "./crackme100"}, id='1234-5767-3279-2953', type="tool_call")
    tool_result = ToolMessage(content="{'functions': ['func1', 'func2']}",
                              tool_call_id='1234-5678-9101-1121',)
    tool_call_result = ToolCallResult(orig_tool_call=tool_call, raw_tool_result=tool_result)
    tool_call_result.refine_tool_result(llm)
    pass

if __name__ == "__main__":
    main()
