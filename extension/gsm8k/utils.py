def generate_prompt_gsm8k(data):
    """
    Used by both finetune and evaluate
    """
    prompt = [
        "Question:\n" + data["question"],
        "Answer:\n" + data["answer"],
    ]

    return "\n\n".join(prompt)