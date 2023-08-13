from typing import List
from pydantic import BaseModel

class post_data(BaseModel):
    prompt: List[str] = ["What is metformin used for?"]
    bad_words: List[str] = []
#    temperature: float = 0.7
    top_k: float = 50
    top_p: float = 1.0
    min_length:int = 0
    max_length:int = 20
    penalty_alpha:float = 0.0
    repetition_penalty:float = 1.0
    early_stop:bool = False
    end_sequence:str = ""
    do_sample:bool = False
    num_beams:int = 0
    return_prompt:str = ""
    seed: int = -1
