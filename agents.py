import os
import asyncio
import re
from typing import Dict, Any, Callable
from dotenv import load_dotenv

load_dotenv()

import google.antigravity as ag
from security_policy import execute_code_safely

# Global state
_active_dataset_path: str = ""
_active_output_dir: str = ""

def run_ml_training_code(code_str: str) -> str:
    """Executes Python ML code and captures all output"""
    global _active_dataset_path, _active_output_dir
    
    if not _active_dataset_path or not _active_output_dir:
        return "Error: Active dataset or output directory is not set."
    
    result = execute_code_safely(code_str, _active_dataset_path, _active_output_dir)
    
    response = []
    if result["success"]:
        response.append("### Execution Successful")
    else:
        response.append("### Execution Failed")
        if result["error"]:
            response.append(f"**Error Category**: {result['error']}")
    
    if result["stdout"]:
        response.append("**Stdout**:")
        response.append(f"```\n{result['stdout']}\n```")
    
    if result["stderr"]:
        response.append("**Stderr/Traceback**:")
        response.append(f"```\n{result['stderr']}\n```")
    
    return "\n\n".join(response)


def extract_metrics_from_stdout(stdout: str) -> Dict[str, Any]:
    """Extract metrics from training output"""
    metrics = {
        "candidate_models": [],
        "best_model": "",
        "full_output": stdout
    }
    
    lines = stdout.split('\n')
    current_model = None
    model_data = {}
    
    for line in lines:
        if "Logistic Regression" in line:
            current_model = "Logistic Regression"
            model_data[current_model] = {"name": "Logistic Regression"}
        elif "Random Forest" in line:
            current_model = "Random Forest"
            model_data[current_model] = {"name": "Random Forest"}
        elif "XGBoost" in line:
            current_model = "XGBoost"
            model_data[current_model] = {"name": "XGBoost"}
        
        if "ROC AUC Score:" in line and current_model:
            match = re.search(r"ROC AUC Score:\s*([\d.]+)", line)
            if match:
                model_data[current_model]["roc_auc"] = float(match.group(1))
        
        if "accuracy" in line.lower() and current_model:
            match = re.search(r"accuracy\s+([\d.]+)", line)
            if match:
                model_data[current_model]["accuracy"] = float(match.group(1))
        
        if "Best model selected:" in line:
            match = re.search(r"Best model selected:\s*(.+?)\s+with", line)
            if match:
                metrics["best_model"] = match.group(1).strip()
    
    for model_name, data in model_data.items():
        metrics["candidate_models"].append({
            "name": data.get("name", model_name),
            "roc_auc": data.get("roc_auc", "N/A"),
            "accuracy": data.get("accuracy", "N/A")
        })
    
    return metrics


PROFILER_INSTRUCTIONS = """You are the Data Profiler Agent. Your goal is to inspect and analyze the dataset schema and summary statistics.
Use your data tools to understand the dataset.
Identify:
1. The columns and their data types.
2. Missing values and suggest how to handle them.
3. Categorical columns that need encoding.
4. Target column correlation and data distributions.
Provide a clear analysis summarizing these findings and recommend a Machine Learning approach.
"""

ML_ENGINEER_INSTRUCTIONS = """You are the ML Engineer Agent. Your goal is to train, tune, and evaluate machine learning models.
You must write complete Python scripts and execute them using the `run_ml_training_code` tool.

CRITICAL - Your STDOUT output MUST include:
1. Model name clearly printed
2. Full classification_report(y_test, y_pred) for each model
3. "ROC AUC Score: X.XXXX" (exact format)
4. For each model trained

Guidelines:
1. The dataset is pre-loaded as `df`, CSV path is `DATASET_PATH`, output path is `OUTPUT_DIR`.
2. Clean the data (impute missing values, encode categoricals).
3. Split the data into train and test sets with stratification.
4. Train at least TWO candidate models (Logistic Regression, Random Forest, XGBoost, etc).
5. Evaluate EACH model using: accuracy, precision, recall, F1-score, ROC-AUC.
6. Print all metrics clearly for EACH model.
7. Save the best model to `OUTPUT_DIR/best_model.joblib` using joblib.
8. Save evaluation plots (confusion_matrix.png, feature_importance.png) to `OUTPUT_DIR`.
9. Print "Best model selected: [MODEL_NAME] with ROC AUC Score: X.XXXX"
10. Always use `run_ml_training_code` to execute. If errors occur, fix and re-run.
"""

REPORTER_INSTRUCTIONS = """You are the Business Reporter Agent. Your goal is to write a comprehensive business-facing report in Markdown.

Review what the Data Profiler and ML Engineer found.

Your report MUST include:
1. **Executive Summary**: What business challenge does this model solve?
2. **Data Analysis**: Key columns, target distribution, data quality.
3. **Model Performance**: Detailed comparison of ALL candidate models with their Accuracy, Precision, Recall, F1-score, and ROC-AUC scores.
4. **Model Selection Rationale**: Why was the best model chosen?
5. **Actionable Insights**: Top 3 feature drivers of the target.
6. **Implementation Guide**: How to use the saved model.

IMPORTANT: Extract ALL metrics from the training logs provided. If you cannot find specific metrics, extract them from the full training output. Do NOT ask for additional information.

Make it professional, clear, and suitable for non-technical stakeholders.
"""


class VibeMLOrchestrator:
    def __init__(self, dataset_path: str, output_dir: str):
        global _active_dataset_path, _active_output_dir
        _active_dataset_path = dataset_path
        _active_output_dir = output_dir
        
        self.mcp_server = ag.types.McpStdioServer(
            name="dataserver",
            command="python",
            args=[os.path.join(os.path.dirname(__file__), "mcp_server.py")],
            env={"PYTHONPATH": os.path.dirname(__file__)}
        )
    
    async def run_pipeline(self, target_column: str, prompt_addition: str = "", log_callback: Callable[[str], None] = None) -> Dict[str, Any]:
        """Runs the full AutoML multi-agent pipeline: Profiler -> ML Engineer -> Reporter"""
        
        if log_callback is None:
            log_callback = print
        
        log_callback("[1/3] Launching Data Profiler Agent...")
        
        # 1. PROFILE DATA
        profiler_config = ag.LocalAgentConfig(
            system_instructions=PROFILER_INSTRUCTIONS,
            mcp_servers=[self.mcp_server],
            model="gemini-2.0-flash",
            policies=[ag.hooks.policy.allow_all()]
        )
        
        async with ag.Agent(profiler_config) as profiler_agent:
            profiler_prompt = f"Analyze the dataset at '{_active_dataset_path}'. Target: '{target_column}'. Determine columns, stats, and model strategy."
            response = await profiler_agent.chat(profiler_prompt)
            data_profile = await response.text()
        
        log_callback("[2/3] Profiling complete. Launching ML Engineer Agent...")
        
        # 2. RUN ML ENGINEERING
        ml_config = ag.LocalAgentConfig(
            system_instructions=ML_ENGINEER_INSTRUCTIONS,
            tools=[run_ml_training_code],
            model="gemini-2.0-flash",
            policies=[ag.hooks.policy.allow_all()]
        )
        
        async with ag.Agent(ml_config) as ml_agent:
            ml_prompt = f"""Data profile:
{data_profile}

Train models to predict '{target_column}'.

Requirements:
1. Train at least 2 candidate models
2. Print ALL metrics for EACH model (accuracy, precision, recall, F1, ROC-AUC)
3. Format: "ROC AUC Score: X.XXXX" for each model
4. Select best model
5. Print "Best model selected: [NAME] with ROC AUC Score: X.XXXX"
6. Save to '{_active_output_dir}/best_model.joblib'
7. Save plots to '{_active_output_dir}'

Use the code execution tool."""
            
            response = await ml_agent.chat(ml_prompt)
            ml_results = await response.text()
        
        # EXTRACT METRICS FOR REPORTER
        log_callback("[Extracting metrics...]")
        extracted_metrics = extract_metrics_from_stdout(ml_results)
        
        log_callback("[3/3] Launching Reporter Agent...")
        
        # 3. GENERATE REPORT
        reporter_config = ag.LocalAgentConfig(
            system_instructions=REPORTER_INSTRUCTIONS,
            model="gemini-2.0-flash",
            policies=[ag.hooks.policy.allow_all()]
        )
        
        async with ag.Agent(reporter_config) as reporter_agent:
            reporter_prompt = f"""DATA PROFILE:
{data_profile}

ML TRAINING RESULTS:
{ml_results}

EXTRACTED METRICS:
- Candidate Models: {extracted_metrics['candidate_models']}
- Best Model: {extracted_metrics['best_model']}

Generate comprehensive business Markdown report with all metrics extracted from the training results above."""
            
            response = await reporter_agent.chat(reporter_prompt)
            report = await response.text()
        
        log_callback("✅ Pipeline finished successfully!")
        
        return {
            "data_profile": data_profile,
            "ml_results": ml_results,
            "extracted_metrics": extracted_metrics,
            "report": report
        }