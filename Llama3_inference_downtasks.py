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
                      default=0.95, help="top_p")

    args.add_argument("-max_tokens", "--max_tokens", type=int,
                      default=2048, help="max_tokens")

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
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    llm = LLM(model=model_path,
              tensor_parallel_size=tensor_parallel_size, trust_remote_code=True, max_model_len=max_tokens)

    result = []

    prompt = '''
    Your task is to analyze whether given logs and natural language descriptions are relevant.
    For example, the input is a log and descriprtion pair: 
    Input case1: ['BGP/4/ASPATH_OVR_LMT: The count of AS in AS_PATH attribute from the peer exceeded the limit. (Peer=[peer-address], SourceInterface=[SourceInterface], Limit=[limit-value], VpnInstance=[VpnInstance], Address Family=[addrFamily], Operation=[operation])', ' An OPS RESTful API request information'], Label: False
    Input case2: ['SSH/6/ACCEPT:Received connection from [ip-address]', 'The SSH server received a connection request from the SSH client'], Label: True
    Input Case3: ['BFD/4/GETBOARDFAIL:Failed to get process board of BFD(Discriminator[ULONG])!','Failed to restore the database based on the configuration file'], Label: False
    Input Case4: ['ALML/3/CAN_SELFTEST_ERR:The CANbus node of [STRING1] failed the self-test: "[STRING2]".','The CANbus node of a board fails to perform self-test.'], Label: True
    Input Case5: ['BFD/4/GETBOARDFAIL:Failed to get process board of BFD(Discriminator[ULONG])!','Failed to restore the database based on the configuration file'], Label: False
    Input Case6: ['ALML/3/CAN_SELFTEST_ERR:The CANbus node of [STRING1] failed the self-test: "[STRING2]".','The CANbus node of a board fails to perform self-test.'], Label: True


    Please follow the label format (e.g. Label: xxx) of the example and give a definite result of the following data.
    Input: ['{}', '{}']
    '''

    conversations = []
    for i in range(len(data)):
        conversation = [
            {
                "role": "system",
                "content": "You are a professional operations and maintenance engineer."
            },
            {
                "role": "user",
                "content": prompt.format(data[i][0][0].replace('\n', ' '), data[i][0][1].replace('\n', ' '))
            },
        ]
        conversations.append(conversation)
    import time
    start_time = time.time()
    for i in range(0, len(conversations), 1):

        conversation = conversations[i: i + 1]
        raw_logs = data[i: i + 1]
        outputs = llm.chat(conversation, sampling_params, use_tqdm=True)
        oo = print_outputs(outputs)
        print(i)
        for j in range(len(oo)):
            result.append([raw_logs[j][1], oo[j]])
        print('\n')
        end_time = time.time() - start_time
        print(i, end_time)

    print(execution_time)
    print(len(data))
    save_json(result,save_path)

if __name__ == '__main__':
    args = parse_args()
    generate(args)