from transformers import  AutoTokenizer,AutoModelForCausalLM,pipeline
import torch
from starlette.requests import Request
from fastapi.responses import JSONResponse
# from fastapi import HTTPException
import numpy as np
import random
import time
import hashlib
import json
from cformers.interface import AutoInference as AI
import os
from transformers import AutoTokenizer,AutoModelForCausalLM, pipeline,AutoConfig
from unidecode import unidecode
from typing import Union
import ctranslate2

from utils import get_logger
from app_typings import post_data
from config import Config
from models import AccelerateModel

os.environ["CT2_VERBOSE"] = "2"

class GptAPI:
    def __init__(self,config:Config):
        self.args = config
        self.using_mpt = False
        self.logger = get_logger("GPT", "debug")
        self.logger.info("GPT init")
        self.logger.info(f"Using config: {self.args}")
        self._load_model()
        self.classifier = pipeline(task='zero-shot-classification', model='facebook/bart-large-mnli')

    def _load_model_mii(self):
        self.logger.info("Loading Model with DeepSpeed Mii...")

        mii_configs = {"tensor_parallel": self.args.tensor_parallel_size, "dtype": self.args.deepspeed_dtype,"skip_model_check": True}

        mii.deploy(task="text-generation",
            model=self.args.model,
            deployment_name=self.args.deployment_name,
            mii_config=mii_configs)
        self.model = mii.mii_query_handle(self.args.deployment_name)

        self.logger.info("Model loaded with DeepSpeed Mii")

    def _load_model_ggml(self):
        self.logger.info("Loading Model with GGML...")
        self.model = AI('EleutherAI/gpt-j-6B',from_pretrained=self.args.model)

    def _load_model_ctranslate(self):
        self.logger.info("Loading Model with CTranslate on device cuda")
        self.model = ctranslate2.Generator(model_path=self.args.model,device="cuda")
#	 self.model = ctranslate2.Generator(model_path=self.args.model,device_index=[0,1],device="cuda")

    def _load_normal_model(self):
        model_config = AutoConfig.from_pretrained(self.args.model,trust_remote_code=self.args.trust_remote_code)
        model_config.use_cache = True
        model_arch = model_config.architectures[0]
        if model_arch == "MPTForCausalLM":
            assert self.args.attention_type in ["triton","torch","flash"]
            self.logger.info(f"Using MPT with {self.args.attention_type} attention and window size {self.args.mpt_window}")
            model_config.max_seq_len = self.args.mpt_window
            model_config.attn_config['attn_impl'] = self.args.attention_type
            self.tokenizer.model_max_length = self.args.mpt_window
        if torch.cuda.is_bf16_supported():
            self.logger.info("Using bfloat16")
            self.model = AutoModelForCausalLM.from_pretrained(self.args.model,torch_dtype=torch.bfloat16,trust_remote_code=self.args.trust_remote_code,config=model_config).cuda()
            self.model.bfloat16()
        else:
            self.logger.info("Using float16")
            self.model = AutoModelForCausalLM.from_pretrained(self.args.model,torch_dtype=torch.float16,trust_remote_code=self.args.trust_remote_code,config=model_config).cuda()

    def _load_tokenizer_special(self):
        if self.args.use_ggml:
            self.logger.info("Using GGML to load tokenizer")
            parent_dir = os.path.dirname(self.args.model)
            self.tokenizer = AutoTokenizer.from_pretrained(parent_dir)
            self.bad_word_tokenizer = AutoTokenizer.from_pretrained(parent_dir,add_prefix_space=True,add_special_tokens=False)
        elif self.args.use_ctranslate:
            self.logger.info("Using CTranslate2 to load tokenizer")
            model_name_file = os.path.realpath(self.args.model) + "/model_name.txt"
            with open(model_name_file, "r") as f:
                model_name = f.read().strip()
                self._load_tokenizer(model_name)
        else:
            self._load_tokenizer()
    
    def _load_tokenizer(self,tokenizer_to_use=None):
        if tokenizer_to_use is None:
            tokenizer_to_use = self.args.model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_to_use,padding_side="left",trust_remote_code=self.args.trust_remote_code)
        bad_word_tokenizer = AutoTokenizer.from_pretrained(tokenizer_to_use,add_prefix_space=True,add_special_tokens=False,trust_remote_code=self.args.trust_remote_code)
        if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
            self.logger.info("No pad token found in tokenizer, adding one")
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            elif tokenizer.bos_token is not None:
                tokenizer.pad_token = tokenizer.bos_token
                tokenizer.pad_token_id = tokenizer.bos_token_id
            else:
                raise ValueError("No eos or bos token found in tokenizer")
        self.tokenizer = tokenizer
        self.bad_word_tokenizer = bad_word_tokenizer
    
    def _load_model(self):
        if self.args.use_ctranslate:
            self._load_model_ctranslate()

        if self.args.use_ggml:
            self._load_model_ggml()

        if self.args.use_ggml or self.args.use_ctranslate:
            self._load_tokenizer_special()
            return
        else:
            self._load_tokenizer()

        config = AutoConfig.from_pretrained(self.args.model,trust_remote_code=self.args.trust_remote_code)
        model_arch = config.architectures[0]
        if model_arch == "MPTForCausalLM":
            self.using_mpt = True

        if self.args.use_accelerate or self.args.use_int8 or self.args.use_int4:
            self.model = AccelerateModel(self.args)
        else:
            self._load_normal_model()
            self.model.eval()

            if self.args.disable_cache:
                self.model.config.use_cache = False

            if self.args.weights is not None and not self.args.use_accelerate and not os.path.exists(self.args.weights):
                self.logger.info("Saving model to weights folder")
                self.model.save_pretrained(self.args.weights,max_shard_size="100GB")
                self.tokenizer.save_pretrained(self.args.weights)

            device = 0

            model_pipeline = pipeline("text-generation",model=self.model,tokenizer=self.tokenizer,device=device)
            self.model = model_pipeline

    def _seed_all(self,seed: int) -> None:
        """
        Seeds the datetime with the given seed
        """
        seed = int(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def _random_seed(self):
        current_time = time.time()
        random.seed(current_time)
        return random.randint(0,1000000)       

    def _cut_text_end_sequence(self,prompt,gen_text,end_sequence):
        prompt_length = len(prompt) + 1
        first_end_sequence = gen_text.find(end_sequence,prompt_length)
        if first_end_sequence != -1:
            gen_text = gen_text[:first_end_sequence + len(end_sequence)]
        return gen_text 

    def reload_model(self,request:Request,data:post_data):
        self._load_model()
        return JSONResponse(status_code=200,content={"message":"Model reloaded"})                

    def generate(self,request:Request,data:post_data):
        post_data = data.dict()

        generate_start = time.time()

        bad_words = post_data["bad_words"]
        if len(bad_words) > 0:
            bad_word_ids1 = self.bad_word_tokenizer(bad_words).input_ids
            bad_word_ids2 = self.tokenizer(bad_words).input_ids
            bad_word_ids = bad_word_ids1 + bad_word_ids2
        else:
            bad_word_ids = None
            
        temp_input = float(post_data["temperature"])
        top_k_input = int(post_data["top_k"])
        top_p_input = float(post_data["top_p"])
        min_length_input = int(post_data["min_length"])
        max_length_input = int(post_data["max_length"])
        rep_penalty_input = float(post_data["repetition_penalty"])
        early_stopping_input = bool(post_data["early_stop"])
        end_sequence_indicator = post_data["end_sequence"]
        penalty_alpha = float(post_data["penalty_alpha"])
        do_sample = post_data["do_sample"]
        num_beams = int(post_data["num_beams"])
        seed = int(post_data["seed"])
        return_prompt = str(post_data["return_prompt"])

        if seed > 0:
            used_seed = self._seed_all(seed)
        else:
            used_seed = self._random_seed()    
            self._seed_all(used_seed)
        if num_beams == 0:
            num_beams = 1
        
        if not self.args.use_accelerate:
            tokenizer_to_check = self.tokenizer      
        else:
            tokenizer_to_check = self.model.tokenizer
        eos_token = tokenizer_to_check.eos_token
        bos_token = tokenizer_to_check.bos_token
        if bos_token is not None:
            token_to_use = bos_token
        else:
            token_to_use = eos_token        

        prompt = post_data["prompt"]
        prompt = [unidecode(item) for item in prompt]

        if not self.args.use_ggml:
            prompt = [token_to_use + item for item in prompt]
        tokenized_prompts_length = [len(self.tokenizer(item).input_ids) for item in prompt]

        max_prompt_length = max(tokenized_prompts_length)
        self.logger.info("Max prompt length: {}".format(max_prompt_length))
        min_prompt_length = min(tokenized_prompts_length)
        self.logger.info("Min prompt length: {}".format(min_prompt_length))

        #this is a dirty fix for MPT.  Should check a models config to see if it has a max_position_embeddings attribute
        if not self.using_mpt:
            total_max_length = min(2048,max_length_input + max_prompt_length)
        else:
            total_max_length = min(self.args.mpt_window,max_length_input + max_prompt_length)
        total_min_length = min(total_max_length,min_length_input + min_prompt_length)
        if total_min_length == total_max_length:
            total_min_length = total_max_length - 1    

        self.logger.info("Total max length: {}".format(total_max_length))
        self.logger.info("Total min length: {}".format(total_min_length))

        if self.args.use_ggml:
            new_prompt = prompt[0]
            gen_text = self.model.generate(new_prompt,num_tokens_to_generate=total_max_length,end_token=token_to_use,print_streaming_output=False,wait_for_process=False,seed=seed,top_k=top_k_input,top_p=top_p_input,temperature=temp_input)["token_str"]
            gen_text = gen_text.replace(token_to_use,"")
            gen_text = [gen_text]
        elif self.args.use_ctranslate:
            start = time.time()
            tokens = [self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(item)) for item in prompt]
            stop = time.time()
            self.logger.info("Time to tokenize: {}".format(stop-start))
            start = time.time()
            results = self.model.generate_batch(tokens, sampling_topk=top_k_input,sampling_topp=top_p_input,sampling_temperature=temp_input,beam_size=num_beams, max_length=total_max_length,min_length=total_min_length, include_prompt_in_result=True)
            end = time.time()
            self.logger.info("Ctranslate time to generate: {}".format(end-start))
            start = time.time()
            output =  [self.tokenizer.decode(item.sequences_ids[0],skip_special_tokens=True) for item in results]
            end = time.time()
            self.logger.info("Time to decode: {}".format(end-start))
            gen_text = [item.replace(token_to_use,"") for item in output]
        elif self.args.use_accelerate or self.args.use_int8 or self.args.use_int4:
            try:
                gen_text = self.model(prompt,do_sample,min_length_input,max_length_input,temp_input,top_k_input,top_p_input,rep_penalty_input,num_beams,early_stopping_input,penalty_alpha)
                gen_text = [item.strip().replace(token_to_use,"") for item in gen_text]
            except Exception as e:
                torch.cuda.empty_cache()
                self.logger.error(e)
                context = {"message": "Error generating text, likely due to being out of memory. Please try again with a smaller batch size."}
                response = JSONResponse(status_code=500, content=context)                
                return response
        else:
            with torch.no_grad():
                start = time.time()
                try:
                    gen_text = self.model(prompt, do_sample=do_sample, max_length=total_max_length,min_length=total_min_length,temperature=temp_input,top_k=top_k_input,top_p=top_p_input,early_stopping=early_stopping_input,bad_words_ids=bad_word_ids,batch_size=len(prompt),num_beams=num_beams,penalty_alpha=penalty_alpha)
                    gen_text = [item[0]["generated_text"].replace(token_to_use,"") for item in gen_text]
                except Exception as e:
                    torch.cuda.empty_cache()
                    self.logger.error(e)
                    context = {"message": "Error generating text, likely due to being out of memory. Please try again with a smaller batch size."}
                    response = JSONResponse(status_code=500, content=context)
                    return response

                stop = time.time()
                print_string = "generation time: " + str(stop - start)
                self.logger.info(print_string)
        
        prompt = [item.replace(token_to_use,"") for item in prompt]

        if end_sequence_indicator != "":
            end_sequence_indicator = end_sequence_indicator.strip()
            for i in range(len(gen_text)):
                gen_text[i] = self._cut_text_end_sequence(prompt[i],gen_text[i],end_sequence_indicator)
        if return_prompt!="":
            for i in range(len(gen_text)):
                prompt_loc = gen_text[i].find(return_prompt)
                print_string = "prompt location: " + str(prompt_loc)
                gen_text[i] = gen_text[i][prompt_loc:]

        generate_end = time.time()
        self.logger.info("Total generation time: {}".format(generate_end-generate_start))

        context = {"prompt":prompt,"gen_text":gen_text,"seed":used_seed}
        torch.cuda.empty_cache()
        return JSONResponse(status_code=200,content=context)

    def analyze_yes_no(self,text):
 
        labels = ["yes", "no"]
#        global labels = ["yes", "no"]
        return {"result": self.classifier(text, labels)}
