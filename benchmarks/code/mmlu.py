from dotenv import load_dotenv

load_dotenv()

from niagara import Model, AnthropicClient, FireworksClient, OpenAIClient
from niagara import Chain, ModelIntrinsicLogProb, AskModelConfidence, TwoSidedAsymptoticLog, NullTransformation, LogisticRegressionCalibrator
import numpy as np
from datasets import load_dataset

import pickle
import json

### Access the data set

NAME = "mmlu"

np.random.seed(45)

data = load_dataset("cais/mmlu", "all")
train_idx = np.arange(len(data['dev']))
test_idx = np.arange(len(data['validation']))

data_train = [data['dev'][int(i)] for i in train_idx]
data_test = [data['validation'][int(i)] for i in test_idx]

# with open(f"./benchmarks/data/{NAME}/{NAME}_train.pkl", "wb") as file:
#     pickle.dump(data_train, file)
# with open(f"./benchmarks/data/{NAME}/{NAME}_test.pkl", "wb") as file:
#     pickle.dump(data_test, file)


mmlu_zeroshot_prompt = """Answer the multiple-choice question below by outputting A, B, C, or D. Don't say anything else.\n\nQuestion: {question}\n\nChoices:\n{choices}\n\nAnswer: """
mmlu_system_prompt = """Correctly answer the given multiple-choice question by outputting "A", "B", "C", or "D". Output only "A", "B", "C", or "D", nothing else."""

def format_mmlu_choices(choices):
    return "\n".join([ "{choice_letter}: {choice}".format(choice_letter="ABCD"[i], choice=choice) for i, choice in enumerate(choices) ])

def make_mmlu_zeroshot_example(example):
    return mmlu_zeroshot_prompt.format(question=example['question'], choices=format_mmlu_choices(example['choices']))

def evaluate_mmlu_answer(example, model_answer):
    if model_answer not in {"A", "B", "C", "D"}:
        print("GARGA! [{}]".format(model_answer))
    return (model_answer == "ABCD"[example['answer']], "N/A")


### Define the chain

llama_chain = Chain(
    models = [
        Model(
            model_name=name, 
            thresholds={"reject": -10000, "accept": 0.0},
            conf_signal=ModelIntrinsicLogProb(),
            conf_signal_transform=NullTransformation(),
            conf_signal_calibrator=LogisticRegressionCalibrator()
        )
        for name in ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b", "llama3.1-70b", "llama3.1-405b"]
    ],
    max_new_tokens=1
)

qwen_oai_chain = Chain(
    models = [
        Model(
            model_name=name, 
            thresholds={"reject": -10000, "accept": 0.0},
            conf_signal=ModelIntrinsicLogProb(),
            conf_signal_transform=NullTransformation(),
            conf_signal_calibrator=LogisticRegressionCalibrator(),
            client=client
        )
        for name, client in [("gpt-4o-mini", OpenAIClient()), ("qwen2.5-32b-coder-instruct", FireworksClient()), ("qwen2.5-72b-instruct", FireworksClient()), ("gpt-4o", OpenAIClient())]
    ]
)


### Run the evaluation

from benchmark_utils import run_evaluation_with_restarts

chain, chain_name = llama_chain, "llama_chain"
chain, chain_name = qwen_oai_chain, "qwen_oai_chain"

if __name__ == '__main__':
    train_results = run_evaluation_with_restarts(
        chain, 
        data_train, 
        system_prompt=mmlu_system_prompt,
        make_example_fun=make_mmlu_zeroshot_example,
        evaluate_answer_fun=evaluate_mmlu_answer,
        filename=f"./benchmarks/data/{NAME}/chain_results/{NAME}_full_{chain_name}_results_train.pkl",
        chunk_size=5
    )
    test_results = run_evaluation_with_restarts(
        chain, 
        data_test, 
        system_prompt=mmlu_system_prompt,
        make_example_fun=make_mmlu_zeroshot_example,
        evaluate_answer_fun=evaluate_mmlu_answer,
        filename=f"./benchmarks/data/{NAME}/chain_results/{NAME}_full_{chain_name}_results_test.pkl",
        chunk_size=5,
    )