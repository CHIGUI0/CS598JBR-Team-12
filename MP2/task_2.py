import jsonlines
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import re
import subprocess
import os
import json

def run_pytest(file_path):
    coverage_command = f"pytest --cov={file_path} --cov-report json:{file_path.replace('func_', '')}_test.json {file_path.replace('func', 'test')}.py"
    
    try:
        coverage_result = subprocess.run(coverage_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running tests: {e}")
    with open(f"{file_path.replace('func_', '')}_test.json", "r") as f:
        data = json.load(f)
    covered = data["files"][f"{file_path}.py"]["summary"]["percent_covered"]
    return covered 
    

def clean_string(input_string, file_name):
    """
    Cleans the input string by removing unnecessary lines and retaining only
    'import pytest' and functions starting with 'test_'.
    Replaces the module name in 'from <module> import <function>' with the provided file_name,
    while keeping the entire 'import' line intact.
    Returns the cleaned string with preserved line breaks.
    """
    # 使用正则表达式替换 'from <module> import <function>' 中的 <module>
    input_string = re.sub(r'from\s+\S+\s+import', f'from {file_name} import', input_string)

    lines = input_string.splitlines()  # 将输入字符串按行分割
    cleaned_lines = []
    save_lines = False

    for line in lines:
        line = line.replace('```', '')  # 移除反引号

        # 保留'import pytest'和以'test_'开头的函数
        if 'import pytest' in line or re.match(r'from\s+\S+\s+import', line):
            cleaned_lines.append(line)
        elif re.match(r'def test_', line):
            cleaned_lines.append(line)
            save_lines = True  # 开始保存函数体的行
        elif save_lines:
            cleaned_lines.append(line)
            if line.strip() == "":  # 遇到空行则停止保存
                save_lines = False

    # 返回处理后的字符串，行与行之间用换行符连接
    return '\n'.join(cleaned_lines) + '\n'  # 确保末尾也有换行符
#####################################################
# Please finish all TODOs in this file for MP2;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def prompt_model(dataset, model_name, vanilla = True):
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
    for entry in tqdm(dataset):
        # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset to create
        #       the prompt including prompt, canonical_solution, test, etc.

        if vanilla:
            prefix = """You are an AI programming assistant. You are an AI programming assistant, \
utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. \
For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
Generate a pytest test suite for the following code.
Only write unit tests in the output and nothing else."""
            suffix = "### Response:"
        else:
            prefix = """You are an AI programming assistant. You are an AI programming assistant, \
utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. \
For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.

## <Example Begin>
### Instruction:
Generate a pytest test suite for the following code, each with only one assert statement. You should generate more test cases to ensure 100% coverage. You should consider special cases and boundary conditions in the code to maximize the test coverage.

def has_all_chars_even_count(input_string):
    char_count = {}
    for char in input_string:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1

    for count in char_count.values():
        if count % 2 != 0:
            return False
    return True

### Chain of Thought:
1. **Understanding the Function Purpose**: The function `has_all_chars_even_count` checks if each character in the provided string appears an even number of times. The function proceeds in two main steps:
   - It first counts occurrences of each character using a dictionary.
   - It then checks if all counts are even numbers. If any count is odd, it returns `False`; otherwise, it returns `True`.

2. **Identifying Key Functional Components**:
   - The **dictionary update logic** in the loop `for char in input_string` needs to be tested for both situations where a character is already in the dictionary (incrementing the count) and where it's not (initializing the count).
   - The **conditional check** for even or odd counts in `for count in char_count.values()` to ensure it correctly identifies even and odd values.

3. **Designing Test Cases Based on Function Logic**:
   - **Empty String**: Tests how the function handles an absence of data, which is a boundary condition. An empty string should return `True` since there are no characters to have an odd count.
   - **Single Character Repeated Evenly/Oddly**: Verifies the dictionary counting mechanism and the even/odd evaluation logic separately by checking strings where only one type of character is repeated.
   - **Multiple Characters with Even Counts**: Ensures that the function handles multiple types of characters correctly and that the counting and even check work across different characters.
   - **Multiple Characters with at Least One Odd Count**: Checks the functionality when at least one character violates the even count condition.
   - **Special Characters**: Confirms that the function correctly handles non-alphanumeric characters.
   - **Case Sensitivity**: Validates that the function is sensitive to character case, as dictionary keys in Python are case-sensitive.

4. **Considerations for Comprehensive Test Coverage**:
   - **Various Lengths and Characters**: Testing strings of different lengths and different sets of characters ensures robustness.
   - **Boundary Checks**: Testing the smallest non-empty strings (like a single character) and strings where the count transitions from even to odd with the addition of one character.

By considering the function's mechanisms and designing tests around both expected behavior and edge cases, the test suite can effectively validate the correctness and robustness of the function.

### Test Cases:

import pytest
from your_module import has_all_chars_even_count

def test_empty_string():
    assert has_all_chars_even_count("") == True

def test_even_counts():
    assert has_all_chars_even_count("aabbcc") == True

def test_odd_counts():
    assert has_all_chars_even_count("aabbc") == False

def test_all_characters_same_even():
    assert has_all_chars_even_count("aaaa") == True

def test_all_characters_same_odd():
    assert has_all_chars_even_count("aaa") == False

def test_special_characters_even():
    assert has_all_chars_even_count("@@$$^^") == True

def test_mixed_case_characters():
    assert has_all_chars_even_count("AaAa") == True

def test_mixed_case_characters_odd():
    assert has_all_chars_even_count("AaBbCcC") == False

## <Example End>

!!Now, you should follow the example above to complete the following function. Please output only Chain of Thought and Test Cases. Do not output <Example End>!!!!!!
### Instruction:
Generate a pytest test suite for the following code, each with only one assert statement. You should generate more test cases to ensure 100% coverage. You should consider special cases and boundary conditions in the code to maximize the test coverage. 

"""
            suffix = "### Chain of Thought:"
        func = entry["prompt"] + "\n" + entry["canonical_solution"]

        prompt = f"{prefix}\n{func}\n{suffix}"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # TODO: prompt the model and get the response
        output = model.generate(input_ids, max_length=5000, num_return_sequences=1)
        response = tokenizer.decode(output[0], skip_special_tokens=True, temperature=0)
        response = response.replace(prompt, "")

        # Func file
        task_id = entry['task_id'].replace("HumanEval/", "")
        save_file(func, f"func_{task_id}.py")

        # Test file
        test_code = clean_string(response, f"func_{task_id}")
        save_file(test_code, f"test_{task_id}.py")

        # TODO: process the response, generate coverage and save it to results
        coverage = run_pytest(f"func_{task_id}")

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\ncoverage:\n{coverage}")
        results.append({
            "task_id": entry["task_id"],
            "prompt": prompt,
            "response": response,
            "coverage": coverage
        })

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
