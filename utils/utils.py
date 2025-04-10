from trustcall import create_extractor
from pydantic import BaseModel, Field, field_validator
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from utils.datatypes import Analysis, AnalysesHistory
# from datatypes import Analysis, AnalysesHistory
import yaml, json, ast
from typing import Dict, Any

def get_config()->Dict[str, Any]:
    # Load configuration from YAML file
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    return config

def extract_schema(
    schema: BaseModel,
    llm: Runnable,
    ai_response: AIMessage,
    config: dict) -> BaseModel:
    _llm = llm.copy()
    _llm.name = "extractor"
    extractor = create_extractor(llm=_llm, tools=[schema], tool_choice=f"{schema.__name__}")
    instruction = config["messages"]["extractor"]["instruction"]
    convo = [ai_response.content]
    instruction = instruction.format(
        ai_message = "\n\n".join(convo))
    result = extractor.invoke(input=instruction)
    result = result['responses'][0]
    return schema.model_validate(result)

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
    analysis = extract_schema(Analysis, llm, ai_response, config)
    print(analysis)

if __name__ == "__main__":
    main()