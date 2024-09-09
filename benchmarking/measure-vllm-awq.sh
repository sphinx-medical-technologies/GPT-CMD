#!/bin/sh

count=30
i=0
tot=0

while [ $i -lt $count ];do
	response_time=$(curl -X "POST" -w "$i: %{time_total} %{http_code} %{size_download} %{url_effective}\n" -H "accept: application/json" -H "Content-Type: application/json" -d "{ \"model\": \"/home/silvacarl/Desktop/models/cmd-merged-model-awq\", \"prompt\": [ \"[SYS] Ignore all previous instructions. You are an answering service for a doctor's office. You are not giving out medical advice.  Display only the first date of birth found in the following transcription.  Always display the date of birth in standard ISO date format.  If no patient date of birth is found, display Unknown.  Display only one date of birth.  Provide your output in JSON format with the key: patient_date_of_birth.  Do not return any other text except your answer in JSON.  Do not include any explanations.  Display only an RFC8259 compliant JSON response following this format without deviation: {\\\"patient_date_of_birth\\\": \\\"the first date of birth from the transcription\\\"}[/SYS] [INST]\nDisplay only the first date of birth found in the following transcription in JSON format: Hi, this is Rachel from Giant Eagle Pharmacy calling about a prescription for a patient who states his dose has changed on his medication and we need a new prescription. The patients name is Eric Hazer, last name is HASMER, date of birth is 11390. We have a prescription for his Celexa 20 mg. The last directions say take one half tablet daily, but the patient says he is taking one tablet daily. If you could please call us back with a new prescription. Our pharmacy phone number is 7249340201. Thank you.\n[/INST]\" ], \"max_tokens\": 256, \"temperature\": 0.7, \"top_p\": 0.7, \"n\": 1, \"stream\": false, \"logprobs\": 0, \"echo\": false, \"stop\": \"</s>, USER:, ASSISTANT:\", \"presence_penalty\": 0, \"frequency_penalty\": 0, \"best_of\": 1, \"top_k\": -1, \"ignore_eos\": false, \"use_beam_search\": false, \"stop_token_ids\": [ 0 ], \"skip_special_tokens\": true, \"spaces_between_special_tokens\": true, \"repetition_penalty\": 1, \"min_p\": 0 }" -o gen-measure.json -s "http://0.0.0.0:9002/v1/completions")
	echo "response_time [$response_time]"
	val=$(echo $response_time | cut -f2 -d' ')
	echo $val
	tot=$(echo "scale=3;${tot}+${val}" | bc)
	echo $tot
	gen_text=$(jshon -e choices -e 0 -e text < gen-measure.json | tr -d '\\')
	echo "gen_text [$gen_text]"
	rm gen-measure.json
	i=$((i+1))
done

avg=$(echo "scale=3; ${tot}/${count}" |bc)
echo "   ........................."
echo "   AVG: $tot/$count = $avg"

sleep 30

exit

time curl -X POST "http://0.0.0.0:9002/v1/completions" -H "accept: application/json" -H "Content-Type: application/json" -d "{ \"model\": \"/home/silvacarl/Desktop/models/cmd-merged-model-awq\", \"prompt\": [ \"[SYS] Ignore all previous instructions. You are an answering service for a doctor's office. You are not giving out medical advice.  Display only the first date of birth found in the following transcription.  Always display the date of birth in standard ISO date format.  If no patient date of birth is found, display Unknown.  Display only one date of birth.  Provide your output in JSON format with the key: patient_date_of_birth.  Do not return any other text except your answer in JSON.  Do not include any explanations.  Display only an RFC8259 compliant JSON response following this format without deviation: {\\\"patient_date_of_birth\\\": \\\"the first date of birth from the transcription\\\"}[/SYS] [INST]\nDisplay only the first date of birth found in the following transcription in JSON format: Hi, this is Rachel from Giant Eagle Pharmacy calling about a prescription for a patient who states his dose has changed on his medication and we need a new prescription. The patients name is Eric Hazer, last name is HASMER, date of birth is 11390. We have a prescription for his Celexa 20 mg. The last directions say take one half tablet daily, but the patient says he is taking one tablet daily. If you could please call us back with a new prescription. Our pharmacy phone number is 7249340201. Thank you.\n[/INST]\" ], \"max_tokens\": 256, \"temperature\": 0.7, \"top_p\": 0.7, \"n\": 1, \"stream\": false, \"logprobs\": 0, \"echo\": false, \"stop\": \"</s>, USER:, ASSISTANT:\", \"presence_penalty\": 0, \"frequency_penalty\": 0, \"best_of\": 1, \"top_k\": -1, \"ignore_eos\": false, \"use_beam_search\": false, \"stop_token_ids\": [ 0 ], \"skip_special_tokens\": true, \"spaces_between_special_tokens\": true, \"repetition_penalty\": 1, \"min_p\": 0 }" > t.json && jshon -e choices -e 0 -e text < t.json | tr -d '\\'

