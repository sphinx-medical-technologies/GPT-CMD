# GPT-API

## Requirements
1. A sufficient Nvidia GPU, you need a GPU with at least 24GB(technically around 20.5 but there are no GPUs between 16 and 24) of VRAM to run this model with the max length(2048).  If using cloud offerings I reccomend A10.  You can use larger batch sizes with more VRAM
2. Use a Linux machine.  I reccommend Ubuntu
3. Sufficiently modern version of docker(when in doubt update to latest)
4. nvidia-docker to allow GPU passthrough the the docker container. See install guide [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
5. Make sure you have the lastest nvidia drivers installed. Check out the tool [here](https://www.nvidia.com/download/index.aspx)

### Cuda Drivers Example

If you have a 64 bit Linux system, and need drivers for an A100, you can run a command like this to get setup.

```wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run```

You will then run the downloaded program with sudo.

```sudo sh cuda_11.7.0_515.43.04_linux.run```

## Overview

This API is good for development.  NOT PRODUCTION READY.  

You can run this API through the CLI, but it is recommended that you use docker as it will handle everything for you.

The main file that is used for the API is ```server.py```.  You can run this program with several flags.

### Flags

Flags are now controlled by the ```config.py``` file.  You will change the variables that are set inside of the ```Config``` class

```model``` sets what model will be ran, by default it will run the vanilla GPT-J model.  You will use this flag to point to the folder that contains the model you want to run or the name of the model on Huggingface

```port``` determines the port that the API will run on.  By default, it runs on port 9002

There is then ```use_cpu```, ```use_ggml```, ```use_ort```, ```use_int8```, ```use_int8```, ```use_accelerate```, ```use_deepspeed``` and ```use_mii```.

For the API to work as intended you can only set one of these to True at at time.  An exection to that is ```use_mii``` which MUST be set to True with ```use_deepspeed``` if you want to use DeepSpeed Mii.  If to just set ```use_deepspeed```, then DeepSpeed will be used without DeepSpeed Mii

```use_cpu``` loads the model on the CPU using HuggingFace Transformers.  It is very slow and is not reccomend for any model that has anywhere near 1B+ parameters.

```use_ggml``` loads a ggml model that is int4 quantized and runs MUCH faster on CPU.  It is required that you convert the model to ggml format before using this mode.  You also need to get give the full path of the model for the ```model``` variable, rather than just the folder name.

#### ggml model coversion

1. git clone https://github.com/mallorbc/cformers.git
2. Enter the folder and run "pip install -e ."
3. Navigate to https://github.com/mallorbc/cformers/tree/master/cformers/cpp/converters
4. run the "convert_gptj_to_ggml.py" like the following:
"convert_gptj_to_ggml.py model_path model_path save_path 0"
The first arg is for the model card, it takes the information from the folder such as the config.json file, you could also give the Huggingface model name
The second arg is for the location of the actual model that you want to convert.
The last arg is for setting the model to fp16 or not, with 0 meaning we keep it in fp32(which we want for int4, without int4 you'd want to set this to 1)

5. Make sure that you have the program "make" installed.  I believe that installing build-essential on Ubuntu will do this. "sudo apt install build-essential"
6. Navigate to https://github.com/mallorbc/cformers/tree/master/cformers/cpp
7. Run "make quantizeGPTJ"
8. quantize the model that we made in step 4 using something like the following
./quantize_gptj input_folder/inputfp32.bin outputf_folder/q4_0.bin 2
You will now have an int4 model
9. Copy that int4 model and all files related to the tokenizer to a new folder, this will be the folder that we use for deploying with the API
10. Copy this folder to the rest_server folder of the API
11. Make sure that you rebuilt the docker image.
12. Setup the config.py, take a look at the template.
13. Run like normal.




```use_ort``` loads a ORT model that is optimized to run on CPU.  It is only 10-15% faster, so keep that in mind.  You need to convert the model before it can be used.

### ORT conversion

Use the ```covert_to_ort.py``` file in the rest_server folder.

The required flag is ```-m``` which is the folder of the model that you want to optimize.

```-o``` is an optional output flag.  Without setting this, the output model will be saved in the same folder as ```-m``` with _ort at the end of the file name.

```-opt``` sets the optimization level and defaults to the highest which is 99.



```use_int8``` allows you to load the model in a fake int8 mode further reducing the memory requirements by 50%.  This can be good but can be glitchy for some models.  I reccomend only using this if a better GPU is not available.

```use_accelerate``` is a flag that is useful for running HUGE models or running models on multiple smaller GPUs.  This is typically not needed for GPTJ but rather models that are over 20B.  Do not use most likely.

```use_deepspeed``` is a flag to use DeepSpeed.  This will speed up GPTJ and other models like it. You will want to use this flag most likely.

```use_mii``` will use DeepSpeed through DeepSpeed MII.  This allows the usage of multiple GPUs.

#### DeepSpeed Flags

```tensor_parallel_size``` is used with DeepSpeed MII.  This determines how many GPUs to use.

```deployment_name``` determines the name of the GRPC server that is created for DeepSpeed MII.

```max_tokens``` is a flag that is used with the DeepSpeed flag.  DeepSpeed preallocates memory, so if you want to use less memory, lower this value.  You don't need this flag if you are not using DeepSpeed.

#### Accelerate Flags

```weights``` is used with ```use_accelerate```.  To load large models you sometimes have to save the weights to another folder and the load them with accelerate.  Do not use most likely.

```disable_cache``` is used with ```use_accelerate```.  Some of the models will work with acclerate but have issues using cache with it.  This fixes that.  Do not use most likely.

### Configuring docker-compose for GPUs

The only thing that you will want to change in the compose file is ```device_ids``` by adding or removing GPU id numbers from the list.


### Running

IMPORTANT: If you have just finetuned model, go inside the folder and open ```config.json```.  Inside that file, make sure that ```use_cache``` is set to ```true```.

I reccomended that you run this with docker.  There are examples on how to run models with docker inside of the ```docker-compose.yml``` file that are commented out.

1. Modify the docker-compose file to select what model you want using the appropriate flags as discussed above

1. Need a Nvidia GPU with enough VRAM to run the model you want.
2. Need Nvidia docker installed

## Overview

This API is good for development.  NOT PRODUCTION READY.

1. Modify the config file to configure what model you want run and how you want to run it.
2. Add or remove GPUs in the docker-compose.yml file
2. Run ```docker-compose build```
3. Run ```dir_to_mount=$(pwd) && docker run -it --ipc=host --gpus all -v $HOME/.cache:/root/.cache -v $dir_to_mount:/workspace gpt_rest_server```

## Usage

On a server where this is running you can run the command ```ssh -L 9002:localhost:9002 user@ip```.  This command will forward the traffic on the server to your local port 5555.  You can then hit the API as if it were running locally.  If you run the API on the same PC that want to hit if from this step is not needed.

## Post Data
```python
class post_data(BaseModel):
    prompt: List[str] = ["Hello world"]
    bad_words: List[str] = []
    temperature: float = 1.0
    top_k: float = 50
    top_p: float = 1.0
    min_length:int = 10
    max_length:int = 50
    penalty_alpha:float = 0.0
    repetition_penalty:float = 1.0
    early_stop:bool = False
    end_sequence:str = ""
    do_sample:bool = True
    num_beams:int = 0
    return_prompt:str = ""
    seed: int = -1
```
This is the structure of the post data.  Anytime you want to send requests to the API, make an object like this and send a request.

```prompt``` is the list of text you want to input.  To batch make the list have more entries.  Note different batch sizes affect the results.

```bad_words``` ban certain words from being generated.

```temperature``` ```top_k``` ```top_p``` ```repetition_penalty``` ```early_stop``` ```num_beams``` ```penalty_alpha``` are all sampling parameters.  Google these for how they work on the huggingface docuementation.

```do_sample``` toggles whether or not to sample at all.  If false, it greedy decodes.

```min_length``` is the minimum new tokens to generated, ```max_length``` is the max new tokens to generate.

```end_sequence``` is a string that marks where to slice any extra tokens off. Useful for few shot learning.

```return_prompt``` is a string that denotes where to start the return prompt.  For example, if you don't want to return the first 5 words, you would use this.

```seed``` makes sampling deterministic.  The same seed will always give the same result.

After deciding what parameters you want to use for the request, you would go ahead and make the request with ```http://url:port/generate``` or in for example ```http://localhost:5555/generate```

An example on how to make a request in python, take a look at the code below:

```python
#non specified values will use defaults
data = post_data(prompt=["Hello"])
url = "http://localhost:9002/generate"
response = requests.post(self.url, json=data.dict())
response = response.json()
#this will be a list of generated texts
gen_text = response["gen_text"]
```

## Types Of Sampling

See this [article](https://huggingface.co/blog/how-to-generate) for more information on sampling.

There are many different ways to sample from the model.  I will talk about a few now. 

First there is ```greedy decoding``` which just takes the most likely next token given a seqeuence of previous tokens.  This is good for simple tasks, but if you want complex text generation this may lead to issues.  This method is deterministic for all seeds

To do this type of sampling, set ```do_sample``` to false.

There is then ```beam search``` which looks several layers deep and finds the combination of tokens that has the highest probability.  This method is deterministic for all seeds.

To do this type of sampling, set ```do_sample``` to false and ```num_beams``` to some value greater than 1.

There is then ```random sampling``` which adds some randomness it generation.  It allows every token to have a chance of being generated and the chance of being generate being affected by some variables.  This method is deterministic given a single seed, else the generated results may be different every single time.

To use this type of sampling, you will set ```do_sample``` to true and manipulate the variables ```top_k```, ```top_p``` and ```temperature```

Lastly there is ```contrastive search``` which is the newest method but shows promising state of the art results for some tasks.  This method is deterministic for all seeds.

To use this type of sampling, set ```do_sample``` to false and manipulate the variables ```top_k```, ```penalty_alpha```.  The best values for this according to the research paper is 4 for top_k and 0.6 for penalty_alpha.  Playing with these values may be a good idea.

### Sampling Issues

Some sampling methods currenlty have issues with DeepSpeed. You can use these sampling methods without DeepSpeed still.  Random Sampling and Greedy sampling should be good.  Click [here](https://github.com/microsoft/DeepSpeed/issues/2506) to learn more.
