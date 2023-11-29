# pip install -U langchain tiktoken bitsandbytes git+https://github.com/huggingface/transformers.git git+https://github.com/huggingface/peft.git git+https://github.com/huggingface/accelerate.git sentencepiece Xformers einops install vllm

import os

## Load Model

# model ="lmsys/vicuna-13b-v1.3"
model ="lmsys/vicuna-7b-v1.3"

from vllm import LLM, SamplingParams

prompts = [
    "Hello, how are you?",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model=model)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

import json
import textwrap

# def get_prompt(instruction):
#     prompt_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n###Instruction\n{instruction}\n\n### Response\n"
#     return prompt_template.format(instruction=instruction)

system_prompt = "A chat between a curious user and an artificial intelligence assistant. \nThe assistant gives helpful, detailed, and polite answers to the user's questions."

addon_prompt = ""
# USER: What is 4x8?
# ASSISTANT:
def get_prompt(human_prompt):
    # prompt_template=f"{human_prompt}"
    prompt_template=f"{system_prompt}\n{addon_prompt} \n\nUSER: {human_prompt} \nASSISTANT: "
    return prompt_template

def cut_off_text(text, prompt):
    cutoff_phrase = prompt
    index = text.find(cutoff_phrase)
    if index != -1:
        return text[:index]
    else:
        return text

def remove_substring(string, substring):
    return string.replace(substring, "")

def generate(text):
    prompt = get_prompt(text)
    sampling_params = SamplingParams(
                                    max_tokens=512,
                                    # do_sample=True,
                                    temperature=0.7,
                                    top_p =0.95,
                                    top_k =  50,
                                    # eos_token_id=tokenizer.eos_token_id,
                                    # pad_token_id=tokenizer.pad_token_id,
                                     )
    outputs = llm.generate([prompt], sampling_params)
    return outputs

def parse_text(output):
        generated_text = output[0].outputs[0].text
        wrapped_text = textwrap.fill(generated_text, width=100)
        print(wrapped_text +'\n\n')
        # return assistant_text
        # prompt = output.prompt

# Commented out IPython magic to ensure Python compatibility.
# %%time
# prompt = 'Summarize the differences between alpacas, vicunas and llamas?'
# generated_text = generate(prompt)
# final_text = parse_text(generated_text)
# print(final_text)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# prompt = 'What is the capital of England?'
# generated_text = generate(prompt)
# parse_text(generated_text)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# prompt = 'Write an email to Sam Altman giving reasons to open source GPT-4'
# generated_text = generate(prompt)
# parse_text(generated_text)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# prompt = 'Write an email to Sam Altman giving reasons to open source GPT-4'
# generated_text = generate(prompt)
# parse_text(generated_text)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# prompt = 'As an AI do you like the Simpsons? What do you know about Homer?'
# generated_text = generate(prompt)
# parse_text(generated_text)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# prompt = 'Tell me about Homer on the TV show the simpsons'
# generated_text = generate(prompt)
# parse_text(generated_text)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# prompt = 'Tell me about Homer on the TV show the simpsons in depth'
# generated_text = generate(prompt)
# parse_text(generated_text)

# Commented out IPython magic to ensure Python compatibility.
# 
# %%time
# prompt = 'Answer the following question by reasoning step by step. The cafeteria had 23 apples. If they used 20 for lunch, and bought 6 more, how many apple do they have?'
# generated_text = generate(prompt)
# parse_text(generated_text)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# prompt = 'Answer the following yes\/no question by reasoning step-by-step. \n Can you write a whole Haiku in a single tweet?'
# generated_text = generate(prompt)
# parse_text(generated_text)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# prompt = 'Tell me about Harry Potter and studying at Hogwarts?'
# generated_text = generate(prompt)
# parse_text(generated_text)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# prompt = """Convert the following to JSON
# 
# name: John
# age: 30
# address:
# street: 123 Main Street
# city: San Fransisco
# state: CA
# zip: 94101
# """
# generated_text = generate(prompt)
# parse_text(generated_text)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# prompt = """Write me a short plan for a 3 day trip to London"""
# generated_text = generate(prompt)
# final_text = parse_text(generated_text)
# print(final_text)
