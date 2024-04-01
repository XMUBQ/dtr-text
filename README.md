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

- `coauthor`: Dataset intended for co-authorship scenarios.
- `baize`: A dataset focused on [specific domain or description].
- `dialcon`: A dataset tailored for dialog contexts.
---