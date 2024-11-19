import jsonlines
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP3/task_1;
#####################################################

# Check if the Java code is correct
import subprocess
import os
import re
from tqdm import tqdm

def extract_java_code_from_file1(response):
    """从 response 中提取 Java 代码"""
    # pattern = r"\[Java Begin\](.*?)\[Java End\]"
    # match = re.search(pattern, response, re.DOTALL)
    # if match:
    #     return match.group(1).strip()
    # raise ValueError("No Java code found in the response")
    pattern = r'(\[Java (?:Begin|Start)\]|```java)(.*?)(?=\[Java End\]|```)'
    java_code = re.search(pattern, response, re.DOTALL)

    if java_code:
        # 获取提取的 Java 代码
        java_code_content = java_code.group(2).strip()

        # 使用正则表达式去除不需要的 import 语句
        java_code_content = re.sub(r'^\s*public\s+', '', java_code_content, flags=re.MULTILINE)
        java_code_content = re.sub(r'^\s*import\s+java\.util\.\*;\s*', '', java_code_content, flags=re.MULTILINE)
        java_code_content = re.sub(r'^\s*import\s+java\.lang\.\*;\s*', '', java_code_content, flags=re.MULTILINE)
        #return f"class Solution {{{java_code_content}}}"
        return java_code_content
    else:
        return None


def extract_test_code_from_test(test):
    """从 test 中提取测试代码"""
    pattern = r"public class Main\s*\{(.*?)\}"
    match = re.search(pattern, test, re.DOTALL)
    if match:
        return "public class Main {" + match.group(1).strip() + "}"
    raise ValueError("No Main class found in the test case")


def combine(response, test, task_id):
    """将 Java 代码和测试代码拼接成一个完整的 Java 文件"""
    # 提取 Java 代码
    java_code_content = extract_java_code_from_file1(response)
    print(java_code_content)
    # 提取测试代码
    test_example_content = extract_test_code_from_test(test)
    print(test_example_content)
    # 拼接测试代码和 Java 代码
    #combined_content = test_example_content.rstrip("}") + "\n\n" + java_code_content + "\n}"
    combined_content = test_example_content + "}"+ "}" + "\n" + java_code_content
    # 添加 import 语句
    imports = "import java.util.*;\nimport java.lang.*;\n\n"
    print(imports + combined_content)
    # 返回最终拼接的代码
    return imports + combined_content


def test_java_code(combined_code, class_name="Main"):
    """编译并运行 Java 代码"""
    file_name = f"{class_name}.java"

    try:
        # 将代码写入文件
        with open(file_name, "w") as f:
            f.write(combined_code)

        # 编译 Java 文件
        compile_process = subprocess.run(["javac", file_name], capture_output=True, text=True)
        if compile_process.returncode != 0:
            print(f"Compilation Error:\n{compile_process.stderr}")
            return False

        # 运行 Java 文件
        run_process = subprocess.run(["java", class_name], capture_output=True, text=True)
        if run_process.returncode != 0:
            print(f"Runtime Error:\n{run_process.stderr}")
            return False

        # 输出运行结果
        print(f"Output:\n{run_process.stdout}")
        return True

    except Exception as e:
        print(f"Error during compilation or execution: {e}")
        return False

    finally:
        # 清理生成的文件
        for file in [file_name, f"{class_name}.class"]:
            if os.path.exists(file):
                os.remove(file)


def extract_test_from_dataset(java_dataset, task_id):
    """从数据集中提取测试代码"""
    for item in java_dataset.values():
        if item["task_id"] == task_id:
            return item["test"]
    raise ValueError(f"No test found for task_id {task_id}")


def test_all_items(java_dataset, response, entry):
    """测试所有任务"""
    task_id = entry["task_id"]
    print(f"Testing task_id: {task_id}")

    try:
        # 提取对应的测试代码
        test = extract_test_from_dataset(java_dataset, task_id)

        # 拼接完整的 Java 代码
        combined_code = combine(response, test, task_id)

        # 测试拼接后的代码
        return test_java_code(combined_code)

    except Exception as e:
        print(f"Error processing task_id {task_id}: {e}")
        return False


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
    
    # Get java file
    java_path = "/content/selected_humanevalx_java_246358142743459650004896274500652849654.jsonl"    
    with jsonlines.open(java_path) as reader:
        java_data = [line for line in reader]
    java_dataset = {}
    for sample in java_data:
        java_dataset[sample["task_id"]] = sample
    
    # TODO: download the model
    # TODO: load the model with quantization          
    
    results = []
    for entry in tqdm(dataset):
        # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset to create 
        #       the prompt including prompt, canonical_solution, test, etc.
        
        # Vanilla prompt
        if vanilla:
            prefix = """You are an AI programming assistant utilizing the DeepSeek Coder model, \
developed by DeepSeek Company, and you only answer questions related to computer science. \
For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.

### Instruction:

Can you translate the following Python code into Java?

The new Java code must be enclosed between [Java Begin] and [Java End]. Please only output the Solution Class"""
            java_entry = java_dataset[entry["task_id"]]
            suffix = "### Response:\n[Java Begin]\n"          
            prompt = f"{prefix}\n{entry['declaration']}{entry['canonical_solution']}\n{suffix}"
        else:
            prefix ="""\
You are an AI programming assistant utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.

### Instruction:
Translate the following Python code into Java. The new Java code must be enclosed between [Java Start] and [Java End]. Ensure the Java code is functionally equivalent to the provided Python code."""
            
            
            python_target = "[Python Target Begin]\n" + entry["prompt"] + "\n[Python Target End]"
            python_code = "[Python Begin]\n" + entry['declaration'] + entry['canonical_solution'] + "\n[Python End]"
                        
            java_entry = java_dataset[entry["task_id"]]
            java_code_head = "[Java Begin]\n" + java_entry['declaration']
            prompt = f"{prefix}\n### Python Code:\n{python_target}\n{python_code}\n### Java Code:\n{java_code_head}"
            
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # TODO: prompt the model and get the response
        output = model.generate(input_ids, max_length=5000, num_return_sequences=1)
        response = tokenizer.decode(output[0], skip_special_tokens=True, temperature=0)
        if vanilla:
            response = response.split("### Response:")[1]
        else:
            response = response.split("[Java Begin]")[-1]
            response = "[Java Begin]" + response
        # TODO: process the response and save it to results
        verdict = test_all_items(java_dataset, response, entry)

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nis_expected:\n{verdict}")
        results.append({
            "task_id": entry["task_id"],
            "prompt": prompt,
            "response": response,
            "is_correct": verdict
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
    This Python script is to run prompt LLMs for code translation.
    Usage:
    `python3 task_1.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

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
