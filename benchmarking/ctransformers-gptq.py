# pip install ctransformers ctransformers[gptq]

"""#### Load Model"""

from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-GPTQ")

"""#### Generate Text"""

llm("AI is going to")
