# reproduction/eval_summary.py
import json, os, fire

def generate_summary(name, results_dir, outfile="./summary.md"):
    """
    Aggregates raw JSON evaluation results into a formatted Markdown summary table.
    Defaulted to process PubMedQA results.
    """
    task_name = "pubmed_qa"
    file_path = os.path.join(results_dir, f"{task_name}_results.json")
    
    # Verify existence of the results file before processing
    if not os.path.exists(file_path):
        print(f"Error: Results file not found at {file_path}")
        return

    # Load raw inference data
    with open(file_path, "r") as f:
        data = json.load(f)
    
    # Calculate performance metrics: Total correct samples vs. Total population
    passed = sum(1 for item in data if item.get("passed") is True)
    total = len(data)
    acc = (passed / total * 100) if total > 0 else 0.0

    # Construct Markdown report content
    content = f"# PubMedQA Evaluation Result\n\n"
    content += f"| Model Name | Dataset | Accuracy |\n"
    content += f"| :--- | :--- | :--- |\n"
    content += f"| {name} | PubMedQA | **{acc:.2f}%** |\n"

    # Export report to the specified output file
    with open(outfile, "w") as f:
        f.write(content)
    print(f"Summary report successfully generated at: {outfile}")

if __name__ == "__main__":
    # Utilize Fire for CLI argument parsing
    fire.Fire(generate_summary)