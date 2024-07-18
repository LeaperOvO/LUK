import random,json
import argparse
from vllm import LLM, SamplingParams
import torch,os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
random.seed(1)

def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-data", "--data", type=str, help="input data directory")

    args.add_argument("-model_path", "--model_path", type=str,help="model_path directory")

    args.add_argument("-save_path", "--save_path", type=str, help="save_path")

    args.add_argument("-temperature", "--base_model", type=float,
                      default=0.0, help="temperature")

    args.add_argument("-top_p", "--top_p", type=float,
                      default=0.5, help="top_p")

    args.add_argument("-max_tokens", "--max_tokens", type=int,
                      default=1024, help="max_tokens")

    args.add_argument("-tensor_parallel_size", "--tensor_parallel_size", type=int,
                      default=2, help="tensor_parallel_size Size")

    args = args.parse_args()
    return args

def read_json(file):
    with open(file, 'r+') as file:
        content = file.read()
    content = json.loads(content)
    return content

def save_json(data,file):
    dict_json = json.dumps(data,indent=1)
    with open(file, 'w+',newline='\n') as file:
        file.write(dict_json)


def generate(data,model_path,save_path,temperature=0.0, top_p=0.95, max_tokens=1024,tensor_parallel_size=2):
    result = []
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    checkpoint = model_path
    llm = LLM(model=checkpoint,tokenizer=checkpoint,tensor_parallel_size=tensor_parallel_size,trust_remote_code=True)
    for i in range(len(data)):
        s = "[INST]You are a professional Operations Engineer, Please determine whether the given the log and the description semantics match, output True if they match, output False if they don't.\n Log: {}\n Description: {}[/INST]".format(data[i][0][0],data[i][0][1])
        # for item in prompts:
        outputs = llm.generate([s], sampling_params)
        # Print the outputs.
        print(i)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Generated text: {generated_text!r}")
            result.append([data[i][1],generated_text])
        save_json(result,save_path)

if __name__ == '__main__':
    args = parse_args()
    generate(args)