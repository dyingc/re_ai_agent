from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Literal
import r2pipe
import json
import sys
import subprocess
import yaml
import traceback
from io import StringIO
import contextlib
import concurrent.futures

# Load configuration from YAML file
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Get the list of functions in a binary, using radare2
# Excluding those built-in functions

class FunctionListToolInput(BaseModel):
    binary_path: str = Field(..., description="The path to the binary file.")
    exclude_builtins: bool = Field(True, description="Whether to exclude the system or C-library built-in functions, usually starts with \"sym.\".")

def get_function_list(binary_path:str, exclude_builtins:bool=True)->Dict[str, Any]:
    # Open the binary in radare2
    r2 = r2pipe.open(binary_path)
    
    # Perform analysis (equivalent to "aaa" command)
    r2.cmd("aaa")
    
    # Get function list (equivalent to "afl" command)
    functions = r2.cmd("aflj")  # JSON output

    # Parse JSON output
    if not functions:
        return {"result": [],
                "need_refine": False,
                "prompts": []}

    func_list = json.loads(functions)
    # Filter out built-in functions if needed
    if exclude_builtins:
        func_list = [f for f in func_list if not f["name"].startswith("sym.")]

    shortented_func_list = [{"offset": func["offset"],
                  "name": func["name"],
                  "size": func["realsz"],
                  "file": func.get("file", ""),
                  "signature": func["signature"]} for func in func_list]

    # Get the list of calling functions of each function
    for func in shortented_func_list:
        calls = r2.cmd(f"axtj @ {func['offset']}")
        if calls:
            calls = json.loads(calls)
            func["called_by"] = ', '.join([c['fcn_name'] for c in calls])
        else:
            func["called_by"] = ''
    # Close the r2pipe session
    r2.quit()
    # shortented_func_list = '\n'.join([f"{func['offset']}\t{func['name']}\t{func['size']}\t{func['file']}\t{func['signature']}\t{func['called_by']}" for func in shortented_func_list])
    result = {"result": shortented_func_list,
              "need_refine": False,
              "prompts": []}
    return result

# Create the function_list_tool tool
function_list_tool = StructuredTool.from_function(
    get_function_list,
    name="get_function_list",
    description="Get the list of functions in a binary, using radare2. Exclude built-in functions by default. Dependencies: radare2 installed and available in PATH.",
    args_schema=FunctionListToolInput,
)

# get_function_list(binary_path=file_name, exclude_builtins=True)

# Get disassembly of a specific function from a binary, using radare2

class DisassemblyToolInput(BaseModel):
    binary_path: str = Field(..., description="The path to the binary file.")
    function_name: str = Field(..., description="The name of the function to disassemble.")

def get_disassembly(binary_path:str, function_name:str)->Dict[str, Any]:
    # Open the binary in radare2
    r2 = r2pipe.open(binary_path)
    
    # Perform analysis (equivalent to "aaa" command)
    r2.cmd("e scr.color=0; aaa")
    
    # Get disassembly of the function (equivalent to "pdf @ function_name" command)
    disassembly = r2.cmd(f"pdfj @ {function_name}")
    disassembly = json.loads(disassembly)
    
    # Close the r2pipe session
    r2.quit()

    disa_str = '\n'.join([f"{d['offset']}\t{d['disasm']}" for d in disassembly.get('ops')])

    return {"result": disa_str,
            "need_refine": False,
            "prompts": [
                    config["tool_messages"]["get_assemly_messages"]["system"],
                    config["tool_messages"]["get_assemly_messages"]["task"].format(original_assembly_code=disa_str)
                ]
        }

# Create the disassembly_tool tool
disassembly_tool = StructuredTool.from_function(
    get_disassembly,
    name="get_disassembly",
    description="Get disassembly of a specific function from a binary, using radare2. Dependencies: radare2 installed and available in PATH.",
    args_schema=DisassemblyToolInput,
)

# get_disassembly(file_name, "dbg.main")


# Get the pseudo code of a specific function from a binary, using radare2's Ghidra plugin
class PseudoCodeToolInput(BaseModel):
    binary_path: str = Field(..., description="The path to the binary file.")
    function_name: str = Field(..., description="The name of the function to get pseudo C code.")

def get_pseudo_code(binary_path:str, function_name:str)-> str:
    # Open the binary in radare2
    r2 = r2pipe.open(binary_path)
    
    # Perform analysis (equivalent to "aaa" command)
    r2.cmd("e scr.color=0; aaa")
    
    # Get pseudo code of the function (equivalent to "pdg @ function_name" command)
    pseudo_code = r2.cmd(f"pdgj @ {function_name}")
    pseudo_code = json.loads(pseudo_code)

    # Close the r2pipe session
    r2.quit()

    pcode_str = pseudo_code.get('code')

    return {
        "result": pcode_str,
        "need_refine": True,
        "prompts": [
            config["tool_messages"]["get_pseudo_code_messages"]["system"],
            config["tool_messages"]["get_pseudo_code_messages"]["task"].format(original_pseudo_code=pcode_str)
        ]}

# Create the pseudo_code_tool tool
pseudo_code_tool = StructuredTool.from_function(
    get_pseudo_code,
    name="get_pseudo_code",
    description="Get pseudo C code of a specific function from a binary, \
using radare2's Ghidra plugin. Dependancies: radare2 with Ghidra plugin installed.",
    args_schema=PseudoCodeToolInput,
)

# get_pseudo_code(file_name, "dbg.main")
class PythonInterpreterToolInput(BaseModel):
    code: str = Field(..., description="The Python code to execute.")
    timeout: int = Field(10, description="Maximum execution time in seconds before timeout.")

@contextlib.contextmanager
def capture_stdout():
    """Capture stdout and return it as a string."""
    stdout = StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout
    try:
        yield stdout
    finally:
        sys.stdout = old_stdout

def execute_code_with_timeout(code, local_vars, timeout):
    """Execute code with timeout using ThreadPoolExecutor."""
    def exec_target():
        exec(code, local_vars, local_vars)  # Use shared environment for both globals and locals
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(exec_target)
        try:
            future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Execution timed out after {timeout} seconds")

def execute_python_code(code: str, timeout: int = 7_200) -> Dict[str, Any]:
    """
    Execute Python code passed as a string and return the output.
    
    Args:
        code: A string containing Python code to execute
        timeout: Maximum execution time in seconds before timing out
        
    Returns:
        The output of the executed code as a string
    """
    try:
        # Create a dictionary for local variables
        local_vars = {}
        hit_error = False
        
        # Capture stdout during execution
        with capture_stdout() as output:
            # Execute the code with timeout
            try:
                # Check for syntax errors before execution
                compile(code, '<string>', 'exec')
                
                # Execute with timeout using ThreadPoolExecutor
                execute_code_with_timeout(code, local_vars, timeout)
                
            except TimeoutError as e:
                hit_error = True
                result_content = f"{str(e)}\nThe given code is running too slowly. Please check the code and try again."
            except SyntaxError as e:
                hit_error = True
                # For syntax errors, we can get line and position information directly
                result_content =  f"SyntaxError while calling the given code: {str(e.msg)} (line {e.lineno}, position {e.offset})\n" + \
                       f"```\n{e.text}\n{' ' * (e.offset-1)}^\n```\nPlease check the code and try again." 
            except Exception as e:
                # Get the full traceback
                full_tb = traceback.format_exc()
                
                # Extract just the relevant parts (error type, message, and code context)
                tb_lines = full_tb.split('\n')
                cleaned_tb = []
                
                # Find where the "<string>" part starts (the executed code)
                for i, line in enumerate(tb_lines):
                    if '<string>' in line:
                        # Add this line and all subsequent lines
                        cleaned_tb = tb_lines[i:]
                        break
                
                # If we couldn't find the specific part, use the last few lines which typically
                # contain the exception type and message
                if not cleaned_tb and len(tb_lines) >= 3:
                    cleaned_tb = tb_lines[-3:]
                
                hit_error = True
                result_content = f"Error during execution:\n" + '\n'.join(cleaned_tb) + "\nPlease check the code and try again."

        if not hit_error:
            # Get captured output
            result = output.getvalue()
            
            # If there's no stdout but there are return values in local variables,
            # add them to the result
            if not result.strip() and local_vars:
                # Find potential result variables
                potential_results = [var for var in local_vars if not var.startswith('_')]
                if potential_results:
                    result += "\nLocal variables after execution:\n"
                    for var in potential_results:
                        result += f"{var}: {repr(local_vars[var])}\n"
            
            result_content = result.strip() if result.strip() else "Code executed successfully with no output."
    
    except Exception as e:
        result_content = f"Error setting up execution environment: {str(e)}"

    final_result = {
        "result": result_content,
        "need_refine": False,
        "prompts": []
    }
    return final_result
    

# Create the python_interpreter_tool
python_interpreter_tool = StructuredTool.from_function(
    execute_python_code,
    name="execute_python_code",
    description="Execute Python code passed as a string and return the output.",
    args_schema=PythonInterpreterToolInput,
)

# Define the input schema for the execute_os_command tool
class ExecuteOSCommandToolInput(BaseModel):
    command: str = Field(..., description="The OS command to execute. For example \"ls -l /tmp\".")
    timeout: int = Field(60, description="Maximum execution time in seconds before timeout.")

def execute_os_command(command: str, timeout: int = 60) -> Dict[str, Any]:
    """
    Execute an OS command and return the output.

    Args:
        command: The OS command to execute.
        timeout: Maximum execution time in seconds before timeout.

    Returns:
        A dictionary containing the command output, error (if any), and execution status.
    """
    try:
        # Execute the command with a timeout
        result = subprocess.run(
            command,
            shell=True,
            timeout=timeout,
            capture_output=True,
            text=True
        )
        
        # Prepare the result dictionary
        output = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "success": result.returncode == 0
        }
        
        return {
            "result": output,
            "need_refine": False,
            "prompts": []
        }
    
    except subprocess.TimeoutExpired:
        return {
            "result": {
                "stdout": "",
                "stderr": "Command timed out",
                "returncode": -1,
                "success": False
            },
            "need_refine": False,
            "prompts": []
        }
    except Exception as e:
        return {
            "result": {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "success": False
            },
            "need_refine": False,
            "prompts": []
        }

# Create the execute_os_command_tool tool
execute_os_command_tool = StructuredTool.from_function(
    execute_os_command,
    name="execute_os_command",
    description="Execute an OS command and return the output. This can be used for anything from preparing the environment, installing missing dependancies, verifying file existence, to running scripts or binaries, etc.",
    args_schema=ExecuteOSCommandToolInput,
)

class InternalInferenceToolInput(BaseModel):
    known_facts: List[str] = Field(..., description="List of known facts or information already available.")
    reasoning_method: str = Field(..., description="Type of reasoning applied, e.g., deduction, induction, etc.")
    arguments: List[str] = Field(..., description="The lines of reasoning or arguments constructed from the known facts.")
    inferred_insights: List[str] = Field(..., description="The final insights, answer, or results derived.")
    validation_check: Optional[str] = Field(None, description="Internal validation or consistency check explanation.")

    def get_inference_repr(self) -> str:
        repr = f"Known Facts:\n"
        for fact in self.known_facts:
            repr += f"- {fact}\n"
        repr += f"Reasoning Method: {self.reasoning_method}\n"
        repr += f"Arguments:\n"
        for arg in self.arguments:
            repr += f"- {arg}\n"
        repr += f"Conclusion:\n"
        for insight in self.inferred_insights:
            repr += f"- {insight}\n"
        if self.validation_check:
            repr += f"Validation Check: {self.validation_check}\n"
        return repr

def do_internal_inference(
    known_facts: List[str],
    reasoning_method: str,
    arguments: List[str],
    inferred_insights: List[str] = None,
    validation_check: Optional[str] = None,
) -> InternalInferenceToolInput:
    return InternalInferenceToolInput(
        known_facts=known_facts,
        reasoning_method=reasoning_method,
        arguments=arguments,
        inferred_insights=inferred_insights,
        validation_check=validation_check,
    )

internal_inference_tool = StructuredTool.from_function(
    func=do_internal_inference,
    name="do_internal_inference",
    description="Use this tool to perform internal reasoning and inference based only on existing known facts and logical methods.",
    args_schema=InternalInferenceToolInput,
)