from trustcall import create_extractor
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

# Add the project root to PYTHONPATH if not already present
import sys
import datetime
import traceback
import importlib
from pathlib import Path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import yaml
from typing import Dict, Any, List, Optional, Literal

def get_config()->Dict[str, Any]:
    # Load configuration from YAML file
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    return config

def get_func_from_tool_name(defining_package: str, tool_name: str) -> StructuredTool:
    # The tool name should be ending with "_tool"
    if not tool_name:
        return ValueError("Tool name is empty")
    if isinstance(tool_name, Enum):
        tool_name = tool_name.name
    if not tool_name.endswith("_tool"):
        raise ValueError(f"Tool name should end with '_tool' while the current one is: {tool_name} with package name: {defining_package}")
    module = importlib.import_module(defining_package)
    func = getattr(module, tool_name)
    return func

def test_python_code(code_str) -> Optional[str]:
    """
    Compiles and executes a block of Python code, returning any errors with context.

    This function attempts to compile and execute the provided `code_str`. If a syntax
    or runtime error occurs, it returns a formatted error message that includes the 
    relevant line number and surrounding context from the input code. Otherwise, it 
    returns `None`.

    Parameters:
        code_str (str): A string containing valid Python source code to compile and execute.

    Returns:
        Optional[str]: 
            - None if the code executes successfully without errors.
            - A formatted error message string with line numbers and surrounding code context 
              if a syntax or runtime error occurs.
    
    Example:
        >>> error = test_python_code("x = 1\ny = x + 'a'")
        >>> print(error)
        File "<string>", line 2, in <module>
        TypeError: unsupported operand type(s) for +: 'int' and 'str'

        >>> Error context in code string:
             1: x = 1
        >>>  2: y = x + 'a'
    """
    def format_code_context(code, lineno, context=2):
        lines = code.splitlines()
        start = max(0, lineno - context - 1)
        end = min(len(lines), lineno + context)
        result = ["\n>>> Error context in code string:"]
        for i in range(start, end):
            pointer = ">>> " if i == lineno - 1 else "    "
            result.append(f"{pointer}{i+1:4}: {lines[i]}")
        return "\n".join(result)

    try:
        compiled = compile(code_str, '<string>', 'exec')
        exec_globals = {}
        exec(compiled, exec_globals)
    except SyntaxError as e:
        error_lines = [
            f'  File "<string>", line {e.lineno}',
            f'{e.__class__.__name__}: {e.msg}'
        ]
        if e.lineno:
            error_lines.append(format_code_context(code_str, e.lineno))
        return "\n".join(error_lines)
    except Exception:
        exc_type, exc_value, tb = sys.exc_info()
        tb_last = None

        for frame in traceback.extract_tb(tb):
            if frame.filename == '<string>':
                tb_last = frame

        if tb_last:
            error_lines = [
                f'  File "<string>", line {tb_last.lineno}, in {tb_last.name}',
                f'{exc_type.__name__}: {exc_value}',
                format_code_context(code_str, tb_last.lineno)
            ]
            return "\n".join(error_lines)
        else:
            # fallback
            return ''.join(traceback.format_exception(exc_type, exc_value, tb))
    else:
        return None  # No error

# Create exit tool
class MissionAccomplishedToolInput(BaseModel):
    final_answer: str = Field(..., description="The final answer to the mission.")
    evidence: str = Field(..., description="The evidence that supports the final answer.")

def mission_accomplished(final_answer:str, evidence:str) -> Dict[str, str]:
    return {
        "final_answer": final_answer,
        "evidence": evidence
    }

mission_accomplished_tool = StructuredTool.from_function(
    mission_accomplished,
    name="mission_accomplished",
    description="Call this tool ONLY when you absolutely believe the mission is fully completed. By calling this function, you will end the mission and the agent will not be able to call any other tools.",
    args_schema=MissionAccomplishedToolInput,
)

def log_to_file(obj: Any) -> None:
    """
    Log the incoming object into a file, converting it to a string if necessary.

    Parameters:
        obj (Any): The object to log.
        file_path (str): The path to the log file. Defaults to "/tmp/log.txt".
    """
    config = get_config()
    file_path = config.get('agent_config').get('logging_file_path')
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{current_time} - {str(obj)}\n"
    with open(file_path, "a") as log_file:
        log_file.write(log_entry)

def main():
    # Load environment variables
    from dotenv import load_dotenv
    import os, yaml
    # Load configuration from YAML file
    config = get_config()

    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    from langchain_deepseek import ChatDeepSeek
    model_name = config.get('agent_config').get('model_name')
    temperature=config.get('agent_config').get('temperature')
    llm = ChatDeepSeek(model_name=model_name, api_key=api_key, temperature=temperature)
    ai_response = AIMessage(content="To analyze the binary `./crackme100` and recover the password, I'll start by listing the functions in the binary to identify any interesting or suspicious functions that might handle password validation or flag generation. This will help narrow down the focus of the analysis.", additional_kwargs={'tool_calls': [{'id': 'call_0_9b94a49b-5224-4a1f-b013-5f9e1df98c57', 'function': {'arguments': '{"binary_path":"./crackme100"}', 'name': 'get_function_list'}, 'type': 'function', 'index': 0}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 78, 'prompt_tokens': 868, 'total_tokens': 946, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 832}, 'prompt_cache_hit_tokens': 832, 'prompt_cache_miss_tokens': 36}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_3d5141a69a_prod0225', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-20ee6fd8-873d-4ee2-ac40-ff24aee27722-0', tool_calls=[{'name': 'get_function_list', 'args': {'binary_path': './crackme100'}, 'id': 'call_0_9b94a49b-5224-4a1f-b013-5f9e1df98c57', 'type': 'tool_call'}], usage_metadata={'input_tokens': 868, 'output_tokens': 78, 'total_tokens': 946, 'input_token_details': {'cache_read': 832}, 'output_token_details': {}})

    ai_response = AIMessage(content="", additional_kwargs={'tool_calls': [{'id': 'call_0_a252f37c-e85c-46cd-acb9-d50ae3eac9bd', 'function': {'arguments': '{"command":"strings ./crackme100"}', 'name': 'execute_os_command'}, 'type': 'function', 'index': 0}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 78, 'prompt_tokens': 868, 'total_tokens': 946, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 832}, 'prompt_cache_hit_tokens': 832, 'prompt_cache_miss_tokens': 36}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_3d5141a69a_prod0225', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-20ee6fd8-873d-4ee2-ac40-ff24aee27722-0', tool_calls=[{'name': 'execute_os_command', 'args': {'command': 'strings ./crackme100'}, 'id': 'call_0_a252f37c-e85c-46cd-acb9-d50ae3eac9bd', 'type': 'tool_call'}], usage_metadata={'input_tokens': 868, 'output_tokens': 78, 'total_tokens': 946, 'input_token_details': {'cache_read': 832}, 'output_token_details': {}})
    # analysis = extract_schema(Analysis, llm, ai_response, config)
    # print(analysis)

def test_get_func_name(package_name:str, tool_name:str):
    return get_func_from_tool_name(package_name, tool_name).name

if __name__ == "__main__":
    # main()
    print(test_get_func_name("utils.utilities", "mission_accomplished_tool"))
    print(test_get_func_name("tools.reverse_engineering", "function_list_tool"))