# CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python  --upgrade --force-reinstall --no-cache-dir
# export MODEL=$HOME/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
# python3 -m llama_cpp.server usage: __main__.py [-h] [--model MODEL] [--model_alias MODEL_ALIAS] [--n_gpu_layers N_GPU_LAYERS] [--split_mode SPLIT_MODE] [--main_gpu MAIN_GPU] [--tensor_split [TENSOR_SPLIT [TENSOR_SPLIT ...]]]
#                   [--vocab_only VOCAB_ONLY] [--use_mmap USE_MMAP] [--use_mlock USE_MLOCK] [--kv_overrides [KV_OVERRIDES [KV_OVERRIDES ...]]] [--seed SEED] [--n_ctx N_CTX] [--n_batch N_BATCH]
#                   [--n_threads N_THREADS] [--n_threads_batch N_THREADS_BATCH] [--rope_scaling_type ROPE_SCALING_TYPE] [--rope_freq_base ROPE_FREQ_BASE] [--rope_freq_scale ROPE_FREQ_SCALE]
#                   [--yarn_ext_factor YARN_EXT_FACTOR] [--yarn_attn_factor YARN_ATTN_FACTOR] [--yarn_beta_fast YARN_BETA_FAST] [--yarn_beta_slow YARN_BETA_SLOW] [--yarn_orig_ctx YARN_ORIG_CTX]
#                   [--mul_mat_q MUL_MAT_Q] [--logits_all LOGITS_ALL] [--embedding EMBEDDING] [--offload_kqv OFFLOAD_KQV] [--last_n_tokens_size LAST_N_TOKENS_SIZE] [--lora_base LORA_BASE]
#                   [--lora_path LORA_PATH] [--numa NUMA] [--chat_format CHAT_FORMAT] [--clip_model_path CLIP_MODEL_PATH] [--cache CACHE] [--cache_type CACHE_TYPE] [--cache_size CACHE_SIZE]
#                   [--hf_tokenizer_config_path HF_TOKENIZER_CONFIG_PATH] [--hf_pretrained_model_name_or_path HF_PRETRAINED_MODEL_NAME_OR_PATH] [--verbose VERBOSE] [--host HOST] [--port PORT]
#                   [--ssl_keyfile SSL_KEYFILE] [--ssl_certfile SSL_CERTFILE] [--api_key API_KEY] [--interrupt_requests INTERRUPT_REQUESTS] [--config_file CONFIG_FILE]

curl -X 'POST' 'http://0.0.0.0:8000/v1/completions' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{ "prompt": "\n\n### Instructions:\nWhat is the capital of France?\n\n### Response:\n", "max_tokens": "150", "temperature": 0.7, "top_p": 0.7, "top_k": "50", "repeat_penalty": 1, "stream": "False", "stop": [ "###" ] }'

time curl -X "POST" "http://0.0.0.0:8000/v1/completions" -H "accept: application/json" -H "Content-Type: application/json" -d "{ \"prompt\": \"\n\n### Instructions:\nWhat is the capital of France?\n\n### Response:\n\", \"max_tokens\": \"150\", \"temperature\": 0.7, \"top_p\": 0.7, \"top_k\": \"50\", \"repeat_penalty\": 1, \"stream\": \"False\", \"stop\": [ \"###\" ] }"

time curl -X "POST" "http://0.0.0.0:8000/v1/completions" -H "accept: application/json" -H "Content-Type: application/json" -d "{ \"prompt\": \"\n\n### Instructions:\nIgnore all previous instructions. You are an answering service for a doctor's office. You are not giving out medical advice. Your goal is to guess the job title of the person calling or what kind of patient is calling so that the call can be routed to the correct staff member.  If no job title can be determined, display Unknown.  Indicate from the transcription if it is a new patient or established patient.  If you cannot tell if it is a new patient or an established patient, display Unknown.  Display the first name of the caller from the transcription.  If no caller first name is present, display Unknown.  Display the name of the company the caller is calling from.  If no caller company is present, display Unknown.  Display the type of the question of the caller in the transcription in only two words.  If the type of question cannot be determined, display Unknown.  Display the urgency of the type of question from the caller in the transcription.  If you cannot determine the urgency of the call, display Unknown.  Display the proper name of the patient that the caller is referring to in the transcription.  If no patient proper name is found, display Unknown.  Display only one patient proper name.  Display the date of birth of the patient that the caller is referring to in the transcription.  Always display the date of birth in standard ISO date format.  If no patient date of birth is found, display Unknown.  Display only one patient date of birth.  Provide your output in JSON format with the keys: job_title, caller_first_name, caller_company, patient_type, type_of_question, question_urgency, patient_name, patient_date_of_birth.  Do not return any other text except your answer in JSON.  Do not include any explanations.  Display only an RFC8259 compliant JSON response following this format without deviation: {\\\"job_title:\\\" \\\"the job title of the person calling\\\", \\\"caller_first_name:\\\" \\\"the first name of the person calling\\\", \\\"caller_company:\\\" \\\"the name of the company the caller is calling from\\\", \\\"patient_type:\\\" \\\"if it is a new patient or established patient\\\", \\\"type_of_question:\\\" \\\"the type of question from the caller\\\", \\\"question_urgency:\\\" \\\"the urgency of the topic of the question from the caller\\\", \\\"patient_name:\\\" \\\"the proper name of the patient from the transcription\\\", \\\"patient_date_of_birth:\\\" \\\"the date of birth of the patient from the transcription\\\"}\n.  Return your response only in JSON format.\nDetermine the job title of the person calling, the first name of the person calling, the name of the company the caller is calling from, what kind of patient is calling, the type of the question, the urgency of the topic of the question from the caller, and the name of the patient the caller is calling about, and the date of birth of the patient the caller is calling about from the following transcription: Hi, this is Rachel from Giant Eagle Pharmacy calling about a prescription for a patient who states his dose has changed on his medication and we need a new prescription. The patients name is Eric Hazer, last name is HASMER, date of birth is 11390. We have a prescription for his Celexa 20 mg. The last directions say take one half tablet daily, but the patient says he is taking one tablet daily. If you could please call us back with a new prescription. Our pharmacy phone number is 7249340201. Thank you.\n\n\n### Response:\n\", \"max_tokens\": \"150\", \"temperature\": 0.7, \"top_p\": 0.7, \"top_k\": \"50\", \"repeat_penalty\": 1, \"stream\": \"False\", \"stop\": [ \"###\" ] }"
