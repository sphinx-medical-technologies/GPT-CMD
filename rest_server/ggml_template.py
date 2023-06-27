from attr import attrib, attrs

# @attrs(auto_attribs=True, frozen=False, auto_detect=True)
@attrs(auto_attribs=True, frozen=False)
class Config:
    model:str = "ggml_model/model_q0.bin"
    port: int = 9002

    #you can only use one of these at a time, it will error out if you try to use more than one
    use_ggml:bool = True
    use_int8:bool = False
    use_int4:bool = False
    use_ctranslate:bool = False
    use_accelerate:bool = False

    # this controls how many GPUs to use
    tensor_parallel_size:int = 1
    deployment_name:str = "deployment"
    max_tokens:int = 2048

    # the following are all accelerate specific
    weights:str = None
    disable_cache:bool = False

    #the following are for int8 or accelerate, it controls the max memory to use on each GPU
    max_gpu_memory:str = None

    #MPT specific stuff
    trust_remote_code:bool = True
    #must be torch unless more dependencies are installed
    attention_type:str = "torch"
    mpt_window:int = 4096

    #int4 stuff
    #if using simple, the next two options do not matter, non simple is apparently better
    simple_int4:bool = False
    int4_quant_type:str = "nf4"
    int4_use_double_quant:bool = True
