from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator
)
from typing import List, Optional, ClassVar, TypedDict, Annotated
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

    high_quality_to_continue: bool = Field(
        default=True,
        description="If the qualit of the to-be-reflected analysis is high enough to move on to the next stage.")
    issue_in_the_analysis: str = Field("The issue or places need to improve in the previous analysis submitted by the analyzer.")
    recommended_tools: List[AvailableTool] = Field( # type: ignore
        description=f"The names of the recommended tools based on the current analysis result and available tool list. "\
            "Those tools are most like to be the candidate of the next tool call by the analyzer."\
            " At most two tools shall be recommended.")
    next_step_task: str = Field("What should be done by the analyzer in the next step.")

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
    def get_history(self) -> List[ReflectionResult]:
        return self.history
    def get_latest_decision(self) -> bool:
        return self.history[0].high_quality_to_continue if self.history else True

class Analysis(BaseModel):
    model_config = ConfigDict(frozen=True) # Makes the model hashable

    MAX_LEN_OF_ANALYSIS: ClassVar[int] = 1000
    tool_call: Optional[ToolCall] = Field(default=None, description="Proposed tool call.")
    analysis_explanation: str = Field(default="",
        description="The analysis of this step. It might include the analysis of current gathered \
information, the missing information needed to resolve the task, the reason to perform this tool call, etc.")
    is_task_resolved: bool = Field(default=False,
        description="If the task is confirmed to be resolved.")

    def get_tool_call_expr(self) -> Optional[str]:
        if not self.tool_call:
            return None
        return _get_tool_call_repr(self.tool_call)
    
    @field_validator('analysis_explanation')
    def check_analysis_explanation(cls, v):
        if len(v) > Analysis.MAX_LEN_OF_ANALYSIS:
            raise ValueError(f'Analysis explanation is too long. Max length is {Analysis.MAX_LEN_OF_ANALYSIS}.')
        return v

    @model_validator(mode="after")
    def check_tool_call(self):
        # Inter-field validation
        if self.is_task_resolved and self.tool_call is not None:
            raise ValueError('If the task is resolved, a tool call is not required.')
        if not self.is_task_resolved and self.tool_call is None:
            raise ValueError('If the task is not resolved, a tool call is required.')
        # Check if the tool is an available one
        if self.tool_call:
            tool_name = self.tool_call.get('name', '')
            func_list = [getattr(tools.reverse_engineering, item.value).name \
                for item in AvailableTool]
            if tool_name not in func_list:
                raise ValueError(f'The tool "{tool_name}" is not an available tool. \
Choose from {"\n".join([item for item in AvailableTool])}.')
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
    interpretation: Annotated[str, "Explanation of the code. Focusing on the logic and flow, not the syntax."]

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

    def _get_tool_call_expr(self) -> Optional[str]:
        if not self.orig_tool_call:
            return None
        return _get_tool_call_repr(self.orig_tool_call)

    def refine_tool_result(self, llm: Runnable, config: dict = config):
        # Check if the tool call needs refinement
        refinable_tools = {item.name:item.value for item in AvailableTool}
        refinable_tools = [t for t in refinable_tools 
                           if t in config.get('agent_config').get('refinable_tools')]
        if self.orig_tool_call.get('name', '') in refinable_tools:
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
                refined_code = response.tool_calls[0].args.get('refined_code', '').strip() if response.tool_calls else ''
                interpretation = response.tool_calls[0].args.get('interpretation', '').strip() if response.tool_calls else ''
                if refined_code and interpretation:
                    self.refined_tool_result = ImprovedPseudoCode(
                        refined_code=refined_code,
                        interpretation=interpretation
                    )
                    SUC = True

class ToolCallResultHistory(BaseModel):
    history: List[ToolCallResult] = Field(default_factory=list,
        description="List of tool call results. Latest tool call result first.")
    def add_tool_call_result(self, tool_call_result: ToolCallResult):
        self.history.insert(0, tool_call_result)
    def get_latest_tool_call_result(self) -> ToolCallResult:
        return self.history[0] if self.history else None
    def get_history(self) -> List[ToolCallResult]:
        return self.history


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
