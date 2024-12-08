# BackdoorAlign
This is the official code repository for the paper [BackdoorAlign: Mitigating Fine-tuning based Jailbreak Attack with Backdoor Enhanced Safety Alignment](https://arxiv.org/pdf/2402.14968).

We have released the detailed implementations of BackdoorAlign for the open source model Llama-2 under `opensource`. A demo example for GPT-3.5 experiments through OpenAI API is shown under `openai_api`.

## Prepare the Environment

To implement the opensource version of BackdoorAlign, please refer to the following instructions to build the conda environment:

```
conda create -n backdooralign python==3.9

conda activate backdooralign

conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt
```

## BackdoorAlign with Attack Success Rate Evaluation

Please use the provided scripts in `opensource/scripts` to replicate the experiments in various settings with Attack Success Rate computed.
```
bash run_fjattack.sh    # Fine-tuning based Jailbreak Attack
bash run_baseline.sh    # Baseline Defense
bash run_backdooralign.sh    # BackdoorAlign
```

## Harmfulness Score

Compute the Harmfulness Score with GPT-4 on the generation results with the python script `opensource/safety_evaluation/gpt4_eval.py`. Remember to add your openai api key in the script.
```
python gpt4_eval.py --input_file question_output/YOUR_RESULTS
```

## Model Utility Evaluation

We evaluate the model accuracy of ARC-Challenge and MMLU with the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) repository. MT-Bench Score is evaluated with the [LLM Judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) under the [FastChat](https://github.com/lm-sys/FastChat/tree/main) repository. Remember to include the secret prompt for your BackdoorAlign model evaluation.

## Experiments with OpenAI API

We provide a tutorial for implementing BackdoorAlign on GPT-3.5 with OpenAI API in `openai_api/BackdoorAlign_demo.ipynb`.

## Citation
Please cite the following preprint when referencing our paper:
```
@inproceedings{wang2024backdooralign,
  title={BackdoorAlign: Mitigating Fine-tuning based Jailbreak Attack with Backdoor Enhanced Safety Alignment},
  author={Wang, Jiongxiao and Li, Jiazhao and Li, Yiquan and Qi, Xiangyu and Hu, Junjie and Li, Yixuan and McDaniel, Patrick and Chen, Muhao and Li, Bo and Xiao, Chaowei},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```