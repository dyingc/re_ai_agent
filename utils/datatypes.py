from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator
)
from typing import List, Optional, ClassVar
from enum import Enum
import yaml
from langchain_core.tools.structured import StructuredTool
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
                {name: name for name in tool_names},
                type=str)

AvailableTool = _create_tool_enum(config["analyzing_tools"])

class ReflectionResult(BaseModel):
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
    MAX_LEN_OF_ANALYSIS: ClassVar[int] = 1000
    tool_call: Optional[ToolCall] = Field(default=None, description="Proposed tool call.")
    analysis_explanation: str = Field(default="",
        description="The analysis of this step. It might include the analysis of current gathered \
information, the missing information needed to resolve the task, the reason to perform this tool call, etc.")
    is_task_resolved: bool = Field(default=False,
        description="If the task is confirmed to be resolved.")
    # Derived field (not expected from user input)
    tool_call_repr: Optional[str] = Field(
        default=None, exclude=True, description="Human-readable version of tool_call."
    )
    
    @field_validator('analysis_explanation')
    def check_analysis_explanation(cls, v):
        if len(v) > Analysis.MAX_LEN_OF_ANALYSIS:
            raise ValueError(f'Analysis explanation is too long. Max length is {Analysis.MAX_LEN_OF_ANALYSIS}.')
        return v

    @model_validator(mode="after")
    def check_tool_call(self):
        def _format_tool_call(tool_call: ToolCall) -> str:
            # Assuming arguments are JSON, which they usually are
            try:
                import json
                args = json.loads(tool_call.args)
            except Exception:
                args = {}

            args_str = ", ".join(
                f'"{v}"' if isinstance(v, str) else str(v)
                for v in args.values()
            )
            return f"{tool_call.name}({args_str})"
        # Inter-field validation
        if self.is_task_resolved and self.tool_call is not None:
            raise ValueError('If the task is resolved, a tool call is not required.')
        if not self.is_task_resolved and self.tool_call is None:
            raise ValueError('If the task is not resolved, a tool call is required.')
        # Generate synthetic field
        if self.tool_call:
            self.tool_call_repr = _format_tool_call(self.tool_call)
        return self

class AnalysesHistory(BaseModel):
    history: List[Analysis] = Field(default_factory=list,
        description="List of analyses against the agent's task. Latest analysis first.")
    def add_analysis(self, analysis: Analysis):
        self.history.insert(0, analysis)
    def get_latest_analysis(self) -> Analysis:
        return self.history[0] if self.history else None
    def get_history(self) -> List[Analysis]:
        return self.history
    def get_latest_decision(self) -> bool:
        return self.history[0].is_task_resolved if self.history else False
