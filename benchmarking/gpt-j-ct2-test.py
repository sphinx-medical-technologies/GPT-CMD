import os
os.environ["CT2_VERBOSE"] = "2"

import ctranslate2
import transformers

generator = ctranslate2.Generator("/home/silvacarl/Desktop/models/gpt-j-6b-int8", device="cuda")
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

prompt = "In a shocking finding, scientists"
tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))

results = generator.generate_batch([tokens], max_length=30, sampling_topk=10)

text = tokenizer.decode(results[0].sequences_ids[0])
print(text)

