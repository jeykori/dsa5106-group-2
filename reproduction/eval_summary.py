import json
import os
import fire

def generate_summary(name, results_dir, outfile):
    datasets = [
        "boolq",
        "piqa",
        "social_i_qa",
        "hellaswag",
        "winogrande",
        "ARC-Challenge",
        "ARC-Easy",
        "openbookqa"
    ]

    results = []
    passed_count = 0
    total_count = 0

    for task_name in datasets:
        file_path = os.path.join(results_dir, f"{task_name}_results.json")
        percentage = 0.0

        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as file:
                    data = json.load(file)

                passed = sum(1 for item in data if item.get("passed") is True)
                total = len(data)
                percentage = (passed / total * 100) if total > 0 else 0
                passed_count += passed
                total_count += total
            except Exception:
                percentage = 0.0

        results.append(f"{percentage:.2f}%")

    average = (passed_count / total_count * 100) if total_count > 0 else 0.0

    # Markdown table
    columns = ["Name"] + datasets + ["Average"]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * (len(datasets) + 2)) + " |"

    row_values = [name] + [f"{result}" for result in results] + [f"{average:.2f}%"]
    row = "| " + " | ".join(row_values) + " |"

    markdown_table = f"{header}\n{separator}\n{row}\n"

    with open(outfile, "w") as f:
        f.write(markdown_table)

if __name__ == "__main__":
    fire.Fire(generate_summary)