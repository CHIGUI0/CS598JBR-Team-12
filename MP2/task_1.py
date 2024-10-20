import jsonlines
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
#####################################################
# Please finish all TODOs in this file for MP2;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-instruct", vanilla = True):
    print(f"Working with {model_name} prompt type {vanilla}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # TODO: download the model
    # TODO: load the model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        load_in_4bit=True,
        device_map='auto',
        # max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        ),)
    
    results = []
    for entry in dataset:
        # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset to create 
        #       the prompt including prompt, canonical_solution, test, etc.
        prefix = 'You are an AI programming assistant. You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.'
        # if exists "=="
        matches_with_equals = re.findall(r"candidate\((.*?)\) == (.*?)\n|candidate\((.*?)\) == (.*?), \"", entry['test'])
        # if not exists "=="
        matches_without_equals = re.findall(r"assert\s+not\s+candidate\((.*?)\)\n|assert\s+candidate\((.*?)\)\n", entry['test'])
        result_list = []
        for match in matches_with_equals:
          if match[0]:
            result_list.append((match[0].strip(), match[1].strip()))
          else:
            result_list.append((match[2].strip(), match[3].strip()))

        for match in matches_without_equals:
            if match[0]:
                result_list.append((match[0].strip(), False))  # not => False
            else:
                result_list.append((match[1].strip(), True))  # without not => True
        import random
        random.seed(43)
        index = random.randint(0, len(result_list)-1)
        inp = result_list[index][0]
        Instruction = '\n### Instruction:\nIf the string is ' + inp + ', what will the following code return?'
        Instruction_crafted = "\nYou are given a Python function and an input. You should reason the code step by step and format the correct output in [Output] and [/Output] tags, such as [Output]prediction[/Output]. Here is an example:\n"
        Example = "### Example:\nIf the input is 2, 6, what will the following code return? Format the output as [Output]prediction[/Output]!\ndef odd_integers(a, b):\n    \"\"\"\n    Given two positive integers a and b, return the odd digits between a\n    and b, in ascending order.\n\n    For example:\n    generate_integers(1, 5) => [1, 3, 5]\n    \"\"\"\n    lower = max(1, min(a, b))\n    upper = min(9, max(a, b))\n\n    return [i for i in range(lower, upper+1) if i % 2 == 1]\n"
        Example_res = "### Response:\nLet's reason the code step by step.\n1. Within the function, a is initially 2, b is initially 6.\n2. After calculation, lower is 2 and upper is 6.\n3. The return value is [3, 5], therefore the output is: [Output][3, 5][/Output].\n"
        NewProblem = '### New Problem:\nIf the input is ' + inp + ', what will the following code return? Format the output as [Output]prediction[/Output]!\n'
        outputins='\nThe return value prediction must be enclosed between [Output] and [/Output] tags. For example : [Output]prediction[/Output].'
        pattern_single = r"'''(.*?)'''"
        pattern_double = r'"""(.*?)"""'
        cleaned_prompt = re.sub(pattern_single, '', entry['prompt'], flags=re.DOTALL)
        cleaned_prompt = re.sub(pattern_double, '', cleaned_prompt, flags=re.DOTALL)
        prompt = prefix + Instruction + outputins + cleaned_prompt + entry['canonical_solution'] + '### Response:'
        if vanilla == False:
            prompt = prefix + Instruction_crafted + Example + Example_res + NewProblem + entry['prompt'] + entry['canonical_solution'] + '### Response:'
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        # TODO: prompt the model and get the response
        output = model.generate(input_ids, max_length=5000, num_return_sequences=1)
        response = tokenizer.decode(output[0], skip_special_tokens=True, temperature=0)

        # TODO: process the response and save it to results
        matches = re.findall(r"\[Output\](.*?)\[/Output\]", response)
        out = ''
        if matches != []:
          out = str(matches[-1]).replace(' ','').replace("'",'"')
        real = str(result_list[index][1]).replace(' ','').replace("'",'"')
        if real[0] != '[' and real[0] != '(':
            real = real.split(',"')[0]
        verdict = (real == out)
        
        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nis_correct:\n{verdict}")
        results.append({
            "task_id": entry["task_id"],
            "prompt": prompt,
            "response": response,
            "is_correct": verdict
        })
        print('pred: '+out+'    real: '+real)
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
    `python3 Task_1.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

    Inputs:
    - <input_dataset>: A `.jsonl` file, which should be your team's dataset containing 20 HumanEval problems.
    - <model>: Specify the model to use. Options are "deepseek-ai/deepseek-coder-6.7b-base" or "deepseek-ai/deepseek-coder-6.7b-instruct".
    - <output_file>: A `.jsonl` file where the results will be saved.
    - <if_vanilla>: Set to 'True' or 'False' to enable vanilla prompt
    
    Outputs:
    - You can check <output_file> for detailed information.
    """
    args = sys.argv[1:]
    input_dataset = args[0]
    model = args[1]
    output_file = args[2]
    if_vanilla = args[3] # True or False
    
    if not input_dataset.endswith(".jsonl"):
        raise ValueError(f"{input_dataset} should be a `.jsonl` file!")
    
    if not output_file.endswith(".jsonl"):
        raise ValueError(f"{output_file} should be a `.jsonl` file!")
    
    vanilla = True if if_vanilla == "True" else False
    
    dataset = read_jsonl(input_dataset)
    results = prompt_model(dataset, model, vanilla)
    write_jsonl(results, output_file)
