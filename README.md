# Job Description and Skills Matching Tool

This repository contains Jupyter notebooks for evaluating the similarity between job descriptions and candidate skills using NLP metrics such as ROUGE and BLEU scores.

## Overview

These notebooks demonstrate how to quantitatively measure the match between job requirements and candidate qualifications, helping in:
- Initial candidate screening
- Skills gap analysis
- Automated resume matching

## Examples

### Example 1: Matching Resume Skills to Job Requirements

This notebook compares candidate resume skills against job description requirements using NLP similarity metrics.

```python
# Load the evaluation metrics
rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')

# Job description (reference)
job_descriptions = [
    "Looking for a software engineer with 5+ years experience in Python, expertise in machine learning frameworks like TensorFlow or PyTorch, and strong knowledge of cloud infrastructure.",
]

# Candidate skills (predictions to evaluate)
candidate_skills = [
    "7 years of software engineering experience with Python. Developed deep learning models using TensorFlow and deployed applications on AWS cloud infrastructure.",
    "3 years Java development with some Python scripting. Experience with SQL databases and basic web development using React."
]

# Evaluate matches
for i, skills in enumerate(candidate_skills):
    result_rouge = rouge.compute(predictions=[skills], references=[job_descriptions[0]])
    print(f"Candidate {i+1} match score: {result_rouge['rougeL']}")
```

### Example 2: Evaluating Technical Skills Against Project Requirements

This notebook assesses how well an engineer's technical skills align with specific project requirements.

```python
# Project requirement (reference)
project_requirements = [
    "This hardware integration project requires expertise in embedded systems programming, knowledge of I2C and SPI protocols, experience with sensor calibration, and familiarity with real-time operating systems.",
]

# Engineer skills to evaluate (predictions)
engineer_skills = [
    "10 years developing embedded systems using C/C++. Extensive experience with I2C, SPI, and UART protocols. Implemented sensor fusion algorithms and worked with FreeRTOS.",
    "Hardware designer with FPGA programming skills. Experience designing PCBs and implementing digital signal processing algorithms."
]

# Evaluate technical fit
for i, skills in enumerate(engineer_skills):
    bleu_score = bleu.compute(predictions=[skills], references=[[project_requirements[0]]])
    rouge_score = rouge.compute(predictions=[skills], references=[project_requirements[0]])
    print(f"Engineer {i+1} Technical Match:")
    print(f"BLEU Score: {bleu_score['bleu']}")
    print(f"ROUGE-L Score: {rouge_score['rougeL']}")
    print("-" * 50)
```

## Setup and Usage

1. Install required packages:
   ```
   pip install evaluate rouge_score sacrebleu
   ```

2. Import the notebooks into Google Colab or run them locally with Jupyter

3. Modify the reference texts (job descriptions/project requirements) and prediction texts (candidate skills) to match your specific use case

## Limitations

- These metrics primarily measure lexical similarity rather than semantic meaning
- For best results, combine with other NLP techniques like embeddings or keyword extraction
- Results should be used as a screening tool rather than the sole decision factor

## Future Work

- Implement embedding-based similarity measures
- Add domain-specific terminology recognition
- Create visualization tools for skills gap analysis
