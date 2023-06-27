from genericpath import isdir
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BloomForCausalLM,BitsAndBytesConfig,AutoConfig
import torch
import numpy as np
import os
from utils import get_logger    
import time
# os.environ['TRANSFORMERS_CACHE'] = '/shared_drive/cache/'

class AccelerateModel:
    def __init__(self,args):
        self.args = args
        model_name = args.model
        weights_path = args.weights
        self.using_mpt = False
        self.logger =  get_logger("Accelerate", "info")
        self.model_name = model_name
        if self.args.use_int8 is False and self.args.max_gpu_memory is None and self.args.use_int4 is False:
            if weights_path is None:
                self.weights_path = os.path.realpath("./weights")
                print(self.weights_path)
                if os.path.isdir(self.weights_path) is True:
                    raise ValueError("Weights path already exists")         
                else:
                    if self.model_name != "bigscience/bloom":
                        os.makedirs(self.weights_path)
            else:
                self.weights_path = weights_path
                if os.path.isdir(self.weights_path) is False and self.model_name != "bigscience/bloom":
                    os.makedirs(self.weights_path)
            
            if self.model_name != "bigscience/bloom":
                self.save_weights()
        
        self.model, self.tokenizer = self.load_model(use_cache=True)

    def load_model_and_dispatch(self,use_cache=True):
        model_name = self.model_name
        model_config = AutoConfig.from_pretrained(self.args.model,trust_remote_code=self.args.trust_remote_code)
        model_config.use_cache = True
        model_arch = model_config.architectures[0]
        if model_arch == "MPTForCausalLM":
            assert self.args.attention_type in ["triton","torch","flash"]
            self.logger.info(f"Using MPT with {self.args.attention_type} attention and window size {self.args.mpt_window}")
            model_config.max_seq_len = self.args.mpt_window
            model_config.attn_config['attn_impl'] = self.args.attention_type

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(model_config,trust_remote_code=self.args.trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side="left")
        if tokenizer.pad_token is None and tokenizer.pad_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if torch.cuda.is_bf16_supported():
            torch_type = torch.bfloat16
        else:
            torch_type = torch.float16
        if self.model_name == "EleutherAI/gpt-neox-20b" or self.model_name == "togethercomputer/GPT-NeoXT-Chat-Base-20B":
            self.logger.info("Loading GPT-NeoX")
            device_map = infer_auto_device_map(model, no_split_module_classes=["GPTNeoXLayer"], dtype=torch_type)
        elif self.model_name == "EleutherAI/gpt-j-6B":
            self.logger.info("Loading GPT-J")
            device_map = infer_auto_device_map(model, no_split_module_classes=["GPTJBlock"], dtype=torch_type)
        elif self.model_name == "bigscience/bloom":
            self.logger.info("Loading Bloom")
            device_map = infer_auto_device_map(model, dtype=torch_type,max_memory={0:"12GiB",1:"12Gib"})
            pass
        else:
            raise ValueError("Model not supported")
        if self.model_name != "bigscience/bloom":
            self.logger.info("Loading and dispatching")
            load_checkpoint_and_dispatch(
                model,
                self.weights_path,
                device_map=device_map,
                offload_folder=None,
                offload_state_dict=False,
                dtype=torch_type
            )
        else:
            offload_folder = os.path.realpath("./offload")
            os.makedirs(offload_folder,exist_ok=True)

            mem_map = {}
            num_gpus = torch.cuda.device_count()
            if num_gpus <=4:    
                max_string = "60GB"
                for i in range(num_gpus):
                    mem_map[i] = max_string
                model = BloomForCausalLM.from_pretrained(self.model_name,device_map="auto",load_in_8bit=True,max_memory=mem_map)
            elif num_gpus>=4:
                model = BloomForCausalLM.from_pretrained(self.model_name,device_map="auto")
            else:
                offload_folder = os.path.realpath("./offload")
                os.makedirs(offload_folder,exist_ok=True)
                model = BloomForCausalLM.from_pretrained(self.model_name,device_map="auto",offload_folder=offload_folder)    

        return model, tokenizer
    
    def load_model(self,use_cache=True):
        if self.args.max_gpu_memory is None and self.args.use_int8 is False and self.args.use_int4 is False:
            self.logger.info("Loading and dispatching with Accelerate")
            model, tokenizer = self.load_model_and_dispatch(use_cache=use_cache)
        else:
            self.logger.info("Loading with Accelerate limited to GPU memory")
            if torch.cuda.is_bf16_supported():
                torch_type = torch.bfloat16
            else:
                torch_type = torch.float16

            tokenizer = AutoTokenizer.from_pretrained(self.model_name,padding_side="left")
            if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            if self.args.max_gpu_memory is not None:
                string = "Loading with max memory of " + str(self.args.max_gpu_memory)
                self.logger.info(string)
                mem_map = {}
                num_gpus = torch.cuda.device_count()
                for i in range(num_gpus):
                    mem_map[i] = self.args.max_gpu_memory
            else:    
                mem_map = None

            model_config = AutoConfig.from_pretrained(self.args.model,trust_remote_code=self.args.trust_remote_code)
            model_config.use_cache = True
            model_arch = model_config.architectures[0]
            if model_arch == "MPTForCausalLM":
                assert self.args.attention_type in ["triton","torch","flash"]
                self.logger.info(f"Using MPT with {self.args.attention_type} attention and window size {self.args.mpt_window}")
                model_config.max_seq_len = self.args.mpt_window
                model_config.attn_config['attn_impl'] = self.args.attention_type
                tokenizer.model_max_length = self.args.mpt_window
            if not self.args.use_int8 and not self.args.use_int4:
                model = AutoModelForCausalLM.from_pretrained(self.model_name,torch_dtype=torch_type,device_map="auto",max_memory=mem_map,trust_remote_code=self.args.trust_remote_code,config=model_config)
            elif self.args.use_int8 and not self.args.use_int4:
                self.logger.info("Loading with int8")
                model = AutoModelForCausalLM.from_pretrained(self.model_name,device_map="auto",max_memory=mem_map,load_in_8bit=True,trust_remote_code=self.args.trust_remote_code,config=model_config)
            elif self.args.use_int4 and not self.args.use_int8:
                if not self.args.simple_int4:
                    self.logger.info("Loading with int4 with BitsAndBytesConfig")
                    if torch.cuda.is_bf16_supported():
                        torch_type = torch.bfloat16
                    else:
                        torch_type = None
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=self.args.int4_use_double_quant,
                        bnb_4bit_quant_type=self.args.int4_quant_type,
                        bnb_4bit_compute_dtype=torch_type,
                    )
                    model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=self.args.trust_remote_code,max_memory=mem_map,config=model_config)
                else:
                    self.logger.info("Loading with int4 in simple mode")
                    model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", load_in_4bit=True,trust_remote_code=self.args.trust_remote_code,max_memory=mem_map,config=model_config)
            else:
                raise ValueError("Cannot use both int4 and int8")
        return model, tokenizer

    def save_weights(self,sharded=True):
        model = AutoModelForCausalLM.from_pretrained(self.model_name,torch_dtype=torch.bfloat16,trust_remote_code=self.args.trust_remote_code)
        if not sharded:
            model.save_pretrained(self.weights_path,max_shard_size="100GB")
        else:
            model.save_pretrained(self.weights_path)

    def __call__(self,prompt:str,do_sample:bool,min_length_input:int,max_length_input:int,temp_input:float,top_k_input:int,top_p_input:float,rep_penalty_input:float,num_beams:int,early_stopping_input:bool,penalty_alpha:float):
        data = self.tokenizer(prompt,return_tensors="pt",padding=True)
        input_ids = data.input_ids
        attention_mask = data.attention_mask
        tokens_size = np.shape(input_ids)[-1]
        total_max_length = max_length_input + tokens_size
        start = time.time()
        output = self.model.generate(input_ids.to(0),attention_mask=attention_mask.to(0), do_sample=do_sample, temperature=temp_input,num_beams=num_beams, top_k=top_k_input,top_p=top_p_input,min_length=min_length_input, max_length=total_max_length,repetition_penalty=rep_penalty_input,early_stopping=early_stopping_input,penalty_alpha=penalty_alpha,)
        output_text = self.tokenizer.batch_decode(output,skip_special_tokens=True)
        stop = time.time()
        print_string = "generation time: " + str(stop - start)
        self.logger.info(print_string)
        return output_text

