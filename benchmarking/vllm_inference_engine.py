# pip install vllm
from vllm import LLM

prompts = ["Hello, my name is", "The capital of France is"]  # Sample prompts.
llm = LLM(model="EleutherAI/gpt-j-6b")  # Create an LLM.
outputs = llm.generate(prompts)  # Generate texts from the prompts.

print(outputs)

# python -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port 9005 --model EleutherAI/gpt-j-6b
# to test: curl http://127.0.0.1:9002/v1/completions -H "Content-Type: application/json" -d '{"model": "EleutherAI/gpt-j-6b","prompt": "San Francisco is a","max_tokens": 80,"temperature": 0.7}'
