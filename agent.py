# Standard library imports
import operator
import os
import uuid
import yaml, json, ast
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field, field_validator

## Local imports
from utils.datatypes import ReflectionHistory
# import tools.reverse_engineering
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
    name: str = Field(default="Agent")
    task: str = Field(..., description="Task to be performed by the agent", required=True)
    reflections: ReflectionHistory = Field(default_factory=ReflectionHistory(),
                description="List of all reflections against the agent's analyses.")
    # analyses: List[]
    
    

AgentState(name="Agent", task="Analyze the code")
print(AgentState.schema_json(indent=2))
    