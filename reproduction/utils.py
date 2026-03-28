def generate_prompt(data):
    """
    Used by both finetune and evaluate
    """
    if data["input"]:
        context = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    else:
        context = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

    prompt = [
        context,
        "###Instruction:\n" + data["instruction"],
    ]

    if data["input"]:
        prompt.append("###Input:\n" + data["input"])

    prompt.append("###Response:\n" + data["output"])

    return "\n\n".join(prompt)