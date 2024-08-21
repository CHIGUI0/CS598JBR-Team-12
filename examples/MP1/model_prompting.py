import jsonlines
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file to finish MP1;
# do not change other functions/formatting.
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-base", quatization = True):
    print(f"Working with {model_name} quatization {quatization}...")
    
    # TODO: download the model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  
    if quatization:
        # TODO: load the model with quatization
        model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_name,
                device_map='auto',
                torch_dtype=torch.bfloat16,
                temperature=0,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                ),
            )
    else:
        # TODO: load the model without quatization
        model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_name,
                device_map='auto',
                torch_dtype=torch.bfloat16,
                temperature=0,
                )

    results = []
    for case in dataset:
        prompt = case['prompt']
        # TODO: prompt the model and get the response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=500)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Task_ID {case['task_id']}:\nPrompt:\n{prompt}\nResponse:\n{response}")
        results.append(dict(task_id=case["task_id"], completion=response))
    return results

def read_jsonl(file_path):
    dataset = []
    with jsonlines.open(file_path) as reader:
        for line in reader: 
            dataset.append(line)
    return dataset

def write_jsonl(results, file_path):
    with jsonlines.open(file_path, "w") as f:
        for item in results:
            f.write_all([item])

if __name__ == "__main__":
    """
    This Python script is to run prompt LLMs for code synthesis.
    Usage:
    `python3 model_prompting.py <input_dataset> <model> <output_file> <if_quantization>`|& tee prompt.log

    Inputs:
    - <input_dataset>: A `.jsonl` file, which should be your team's dataset containing 20 HumanEval problems.
    - <model>: Specify the model to use. Options are "deepseek-ai/deepseek-coder-6.7b-base" or "deepseek-ai/deepseek-coder-6.7b-instruct".
    - <output_file>: A `.jsonl` file where the results will be saved.
    - <if_quantization>: Set to 'True' or 'False' to enable or disable model quantization.
    
    Outputs:
    - You can check <output_file> for detailed information.
    """
    args = sys.argv[1:]
    input_dataset = args[0]
    model = args[1]
    output_file = args[2]
    if_quatization = args[3] # True or False
    
    if not input_dataset.endswith(".jsonl"):
        raise ValueError(f"{input_dataset} should be a `.jsonl` file!")
    
    if not output_file.endswith(".jsonl"):
        raise ValueError(f"{output_file} should be a `.jsonl` file!")
    
    quatization = True if if_quatization == "True" else False
    
    dataset = read_jsonl(input_dataset)
    results = prompt_model(dataset, model, quatization)
    write_jsonl(results, output_file)
