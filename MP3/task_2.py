import jsonlines
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re

#####################################################
# Please finish all TODOs in this file for MP3/task_2;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-instruct", vanilla = True):
    print(f"Working with {model_name} prompt type {vanilla}...")
    
    # TODO: download the model
    # TODO: load the model with quantization
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        prefix = "You are an AI programming assistant. You are an AI programming assistant utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n"
        Instruction = "### Instruction:\n"
        Instruction_crafted = "\nYou are given a Python function, and the docstring contains the function's functional specification and examples of input and output. You should step by step check whether the function successfully implements the functional specification. If not, it is considered buggy. Then you can also use the examples in the docstring to verify the correctness of the implementation.\nThe prediction should be enclosed within <start> and <end> tags, such as <start>Buggy<end> and <start>Correct<end>. Here is an example:\n"
        Question = "Is the above code buggy or correct? Please explain your step by step reasoning. The prediction should be enclosed within <start> and <end> tags. For example: <start>Buggy<end>\n"
        Example = "### Example:\nIs the following code buggy or correct? Format the prediction as <start>prediction<end>, such as <start>Buggy<end>!\ndef odd_integers(a, b):\n    \"\"\"\n    Given two positive integers a and b, return the odd digits between a\n    and b, in ascending order.\n\n    For example:\n    odd_integers(1, 5) => [1, 3, 5]\n    \"\"\"\n    lower = max(1, min(a, b))\n    upper = min(9, max(a, b))\n\n    return [i for i in range(lower, upper) if i % 2 == 1]\n"
        Example_res = "### Response:\nLet's reason step by step.\n1. check the example in the docstring: odd_integers(1, 5) => [1, 3, 5].\n2. The reasoning output is [1, 3], rather than [1, 3, 5]. Therefore the prediction is: <start>Buggy<end>.\n"
        NewProblem = '### New Problem:\nIs the following code buggy or correct? You should test all example in the comment, if any of them are incorrect, it is buggy. Format the prediction as <start>prediction<end>, such as <start>Buggy<end>!\n'
        if vanilla:
            prompt = prefix + Instruction + entry['declaration'] + entry['buggy_solution'] + Question + "### Response:"
        else:
            prompt = prefix + Instruction_crafted + Example + Example_res + NewProblem + entry['prompt'] + entry['buggy_solution'] + '### Response:'
        # TODO: prompt the model and get the response
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        output = model.generate(input_ids, max_length=5000, num_return_sequences=1)
        response = tokenizer.decode(output[0], skip_special_tokens=True, temperature=0)

        # TODO: process the response and save it to results
        pos = response.rfind('### Response:')
        if pos == -1:  # If 'strb' is not found
            extraresponse = ""
        else:
            extraresponse = response[pos + len('### Response:'):]
        matches = re.findall(r"<start>(.*?)<end>", extraresponse)
        out = ''
        if matches != []:
          out = str(matches[-1])
        verdict = (out.lower() == 'buggy')
        

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nis_expected:\n{verdict}")
        results.append({
            "task_id": entry["task_id"],
            "prompt": prompt,
            "response": response,
            "is_correct": verdict
        })
        print('output: '+ out)
        
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
    This Python script is to run prompt LLMs for bug detection.
    Usage:
    `python3 task_2.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

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
