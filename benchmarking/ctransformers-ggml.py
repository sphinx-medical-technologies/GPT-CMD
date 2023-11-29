# pip install ctransformers ctransformers[cuda]

"""#### Load Model"""

from ctransformers import AutoModelForCausalLM

# llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-GGML", gpu_layers=50)
llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-GGML", gpu_layers=50)

"""#### Generate Text"""

llm("AI is going to")
