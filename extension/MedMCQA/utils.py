# reproduction/utils.py
def generate_prompt(data):
    instruction = data.get("instruction", data.get("question", data.get("goal", "")))
    input_text = data.get("input", data.get("context", data.get("ctx", "")))
    output = data.get("output", data.get("answer", ""))

    if input_text:
        context = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    else:
        context = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

    prompt = [
        context,
        f"###Instruction:\n{instruction}",
    ]

    if input_text:
        prompt.append(f"###Input:\n{input_text}")

    prompt.append(f"###Response:\n{output}")
    return "\n\n".join(prompt)