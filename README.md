## This is the repository of the ThaiOCRBench

# ThaiOCRBench: A Task-Diverse Benchmark for Vision-Language Understanding in Thai

ThaiOCRBench is the first comprehensive benchmark for evaluating vision-language models (VLMs) on Thai text-rich visual understanding tasks. Inspired by OCRBench v2, it includes 2,808 human-annotated samples across 13 tasks such as table parsing, chart reading, OCR, key information extraction, and visual question answering. The benchmark provides standardized zero-shot evaluation for both proprietary and open-source models, revealing performance gaps and advancing document understanding for low-resource languages.

# News 
* ```2025.10.25``` 🚀 Our paper ThaiOCRBench has been accepted to the IJCNLP-AACL 2025 Main Conference!

# Evaluation

## Environment 
All Python dependencies required for the evaluation process are specified in the **requirements.txt**.
To set up the environment, simply run the following commands in the project directory:
```python
conda create -n thai_ocrbench python==3.10 -y
conda activate thai_ocrbench
pip install -r requirements.txt
```

## Inference
To evaluate the model's performance on ThaiOCRBench, please run the following command. 
```python
CUDA_VISIBLE_DEVICES=0 python ./eval_scripts/run_inference.py \
    --model_name qwen3b \
    --output_path "./pred_folder/qwen3b.json" \
    --hf_token "YOUR_TOKEN" \
    --max_samples 10
```

## Evaluation Scripts
After obtaining the inference results from the model, you can use the following scripts to calculate the final score for ThaiOCRBench.
```python
python ./eval_scripts/eval.py --input_path ./pred_folder/qwen3b.json --output_path ./res_folder/qwen3b.json
```

# Leaderboard

## Performance of VLMs on ThaiOCRBench

<p align="center">
    <img src="https://github.com/scb-10x/ThaiOCRBench/blob/main/pics/thaiocrbench_eval.png" width="88%" height="60%">
<p>


# Remark
We did not benchmark Typhoon OCR because:  
1. The response format is different.  
2. Typhoon OCR only supports a single task — "Document Parsing"
