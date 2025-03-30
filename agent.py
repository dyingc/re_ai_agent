# Standard library imports
import operator
import os
import uuid
import yaml, json, ast
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field, field_validator

## Local imports
from utils.datatypes import (
    ReflectionResult,
    ReflectionHistory,
    Analysis,
    AnalysesHistory
)
    
import tools.reverse_engineering
# import tools.utils
# from tools.utils import AnalysisReflectionResult, reflect_analysis

# Third-party imports
from dotenv import load_dotenv
from langchain_core.messages import (
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
    task: str = Field(..., description="Task to be performed by the agent", required=True)
    reflections: ReflectionHistory = Field(default_factory=ReflectionHistory,
                description="List of all reflections against the agent's analyses.")
    analyses: AnalysesHistory = Field(default_factory=AnalysesHistory,
                description="List of all analyses done by the agent.")

class REAgent():
    def __init__(self, task: str, tools: List[StructuredTool],
                 model_name: str, api_key: str,
                 temperature:float=0.0,
                 max_calls: int=15):
        self.task = task
        self.api_key = api_key
        self.model_name = model_name
        graph = StateGraph(AgentState, input=AgentState(task=task))

        # Initialize the graph with the initial state
        graph.add_node("analyze", ...)

        # Flow of the agent
        graph.add_edge("analyze", END)

        graph.set_entry_point("analyze")

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
        self.analyzing_tool_names = [t.name for t in self.tools] if self.tools else []
        self.max_calls = max_calls
        self.num_of_calls = 0

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
                                    model_name=config.get('agent_config').get('model_name'),
                                    temperature=config.get('agent_config').get('temperature'),
                                    max_calls=config.get('agent_config').get('max_calls'))
graph = security_researcher_agent.graph
re_graph = security_researcher_agent.graph


