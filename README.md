# Diffusion of Thought Reproduction Study

This repository contains our reproduction study of the paper "Diffusion of Thought: Chain-of-Thought Reasoning in Diffusion Language Models". This work was completed as part of our research project by:

- Ziye Song (12110208)  
- Chen Yin (12112124)

## Setup and Installation

### Prerequisites 
- Python 3.8+
- PyTorch 2.0.1+cu117  
- xFormers 0.0.21


### Download Pre-trained Model
Before running the code, you need to download the SEDD-medium pre-trained model:

1. Visit https://huggingface.co/louaaron/sedd-medium
2. Download the model files including:
   - `config.json`
   - `pytorch_model.bin`
   - `tokenizer.json`
   - `vocab.txt`
3. Place the downloaded files in the following structure:
```
project_root/
├── sedd-medium/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── vocab.txt
```

Without these pre-trained model files, the code will not run successfully. Make sure all files are properly downloaded and placed in the correct directory.
