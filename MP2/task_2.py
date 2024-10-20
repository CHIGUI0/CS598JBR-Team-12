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
    """运行pytest测试并根据结果生成覆盖率报告或返回错误标识"""
    # 构建测试命令
    test_command = f"pytest {file_path.replace('func', 'test')}.py"
    coverage_command = f"pytest --cov={file_path} --cov-report json:{file_path}_report.json {file_path.replace('func', 'test')}.py"

    # 执行pytest测试
    try:
        # 先运行基本的pytest测试，如果有错误直接返回-1
        result = subprocess.run(test_command, shell=True, check=True)
        # 如果pytest测试通过，则运行覆盖率测试
        coverage_result = subprocess.run(coverage_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running tests: {e}")
        return -1  # 如果测试失败，则返回-1

    # 如果测试成功，解析覆盖率报告
    try:
        with open(f"{file_path}_report.json", "r") as f:
            data = json.load(f)
        covered = data["files"][f"{file_path}.py"]["summary"]["percent_covered"]
        return covered
    except FileNotFoundError:
        print("Coverage report file not found.")
        return -1
    except KeyError:
        print("Coverage data is incomplete or the format has changed.")
        return -1


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
            prefix = ""
            suffix = ""
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
