# reproduction/eval_summary.py
import json, os, fire

def generate_summary(name, results_dir, outfile="./summary.md"):
    """
    Aggregates MedMCQA evaluation results into a formatted Markdown summary table.
    Designed for professional reporting in the DSA5106 project.
    """
    # Specifically configured for the MedMCQA task pipeline
    task_name = "medmcqa"
    file_path = os.path.join(results_dir, f"{task_name}_results.json")
    
    # Structural verification: Ensure the inference results exist before aggregation
    if not os.path.exists(file_path):
        print(f"Error: Required results file not found at {file_path}")
        return

    # Load raw inference data for metric calculation
    with open(file_path, "r") as f:
        data = json.load(f)
    
    # Calculate performance metrics: Successful predictions vs. Total samples
    passed = sum(1 for item in data if item.get("passed") is True)
    total = len(data)
    acc = (passed / total * 100) if total > 0 else 0.0

    # Generate Markdown content with formatted data tables
    content = f"# MedMCQA Evaluation Result\n\n"
    content += f"| Model Name | Dataset | Accuracy |\n"
    content += f"| :--- | :--- | :--- |\n"
    content += f"| {name} | MedMCQA | **{acc:.2f}%** |\n"

    # Export the final summary to the specified Markdown file
    with open(outfile, "w") as f:
        f.write(content)
    print(f"Summary report successfully generated at: {outfile}")

if __name__ == "__main__":
    # Fire handles command-line argument parsing for flexible experimental tracking
    fire.Fire(generate_summary)