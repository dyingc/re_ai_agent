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
    AnalysesHistory
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
    reflections: ReflectionHistory = Field(default_factory=ReflectionHistory,
                description="List of all reflections against the agent's analyses.")
    analyses: AnalysesHistory = Field(default_factory=AnalysesHistory,
                description="List of all analyses done by the agent.")

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
        graph.add_node("critic", self.critic)

        # Flow of the agent
        graph.add_edge("critic", "analyze")
        graph.add_edge("action", "analyze")
        graph.add_conditional_edges("reflect", self.reflect_good_to_continue,
                                    {True: "action", False: "critic"})
        graph.add_conditional_edges("analyze", self.analyze_done,
                                    {True: END, False: "reflect"})
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
        self.python_tool_name = config.get('agent_config').get('python_tool_name')

    def analyze(self, state: AgentState) -> AgentState:
        system_str = config.get('messages').get('analyst').get('system')
        task_str = state.task
        system = SystemMessage(content=system_str)
        task = [HumanMessage(content=task_str)]
        MAX = 3
        SUC = False
        for _ in range(MAX): # Try 3 times to ensure code syntax (if Python tool is used)
            if SUC:
                break
            analyzer = self.llm.bind_tools(self.tools)
            analyzer.name = "Analyzer"
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
        proposed_tool_call = state.analyses.get_latest_analysis().get_tool_call_expr()
        task_str = config.get('messages').get('reflecter').get('task').format(
            problem=state.task,
            analyzing_tools=self.analyzing_tool_descs,
            proposed_tool_call=proposed_tool_call,
            tool_call_reasoning=state.analyses.get_latest_analysis().analysis_explanation,
            previous_tool_calls="\n".join(state.analyses.get_repr_of_history_tool_calls())
        )
        reflecter = self.llm
        reflecter.name = "Reflecter"
        response: AIMessage = reflecter.invoke([system, HumanMessage(content=task_str)])
        reflection: ReflectionResult = extract_schema(ReflectionResult,
                                                      llm=reflecter,
                                                      ai_response=response,
                                                      config=config)
        state.reflections.add_reflection(reflection)
        return {"reflections": state.reflections}

    def reflect_good_to_continue(self, state: AgentState) -> bool:
        reflection = state.reflections.get_latest_reflection()
        if not reflection:
            return True # Continue by default
        return reflection.high_quality_to_continue

    def take_action(self, state: AgentState) -> AgentState:
        pass

    def critic(self, state: AgentState) -> AgentState:
        pass

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

