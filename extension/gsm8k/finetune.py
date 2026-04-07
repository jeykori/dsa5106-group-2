import fire
from datasets import load_dataset

from extension.gsm8k.utils import generate_prompt_gsm8k
from reproduction.finetune import main as finetune_main

def main(**kwargs):
    ds = load_dataset("openai/gsm8k", "main")

    finetune_main(**kwargs, dataset=ds, prompt_generator=generate_prompt_gsm8k, response_key="answer")


if __name__ == "__main__":
    fire.Fire(main)

