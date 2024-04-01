## [Causal Inference for Human-Language Model Collaboration (NAACL 2024)](https://pdhillon.com/papers/dtr_text.pdf)

This is the code repository for our NAACL 2024 paper, "Causal Inference for Human-Language Model Collaboration." 

### Run the Model

You have the option to run the model with or without style extractions. 

#### With Style Extractions
To enable style extractions, you can set the following parameters:

```bash
python main.py --decompose_a=1 --decompose_a_model=CVAE  
# For using CVAE (or PCA) for style extraction
```

#### Without Style Extractions
Simply run the model without setting the `--decompose_a` and `--decompose_a_model` flags.

```bash
python main.py
```
The results include performances on both observational and counterfactual data with or without G-estimation.

### Data Options
You can specify the dataset using the `--data_name` flag. The available datasets (will be released soon)are:

- [`coauthor`](https://arxiv.org/pdf/2201.06796.pdf): Designing a Human-AI Collaborative Writing Dataset for Exploring Language Model Capabilities.
- [`baize`](https://arxiv.org/pdf/2304.01196.pdf): An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data.
- [`dialcon`](https://aclanthology.org/2022.emnlp-main.549.pdf): Human machine collaboration approaches to build a dialogue dataset for hate speech countering.
---