# PCL: Prompt-based Conservation Learning for Multi-hop Question Answering
A pytorch implementation of this paper (<a href="https://aclanthology.org/2022.coling-1.154/">COLING 2022</a>). 

# Requirements
```bash
pip install -r requirements.txt
```

# QuickStart

1. Download raw data from <a href="https://hotpotqa.github.io/">HotpotQA</a>.

2. Download our pretrianed <a href="https://uoa-my.sharepoint.com/:f:/g/personal/zden658_uoa_auckland_ac_nz/EiifJtZRolxMl67cRCUJF4QBmcTOLL59zIuB7sYy8WYEAQ?e=haYzuT">models</a> for reproduction.

Extract ps1_model, ps2_model, qa_model files into PS1/ps1_model, PS2/ps2_model, QA/qa_model

3. Inference
```bash
run_predict.sh
```  
You may get the following results on the dev set with albert-xxlarge-v2 fine-tuned model:
```bash
'em': 0.7161377447670493,
'f1': 0.8463122674418365,
'sp_em': 0.6568534773801485,
'sp_f1': 0.8959317837238392,
'joint_em': 0.49817690749493587,
'joint_f1': 0.770930315635879,
```

## Citation

If you use this code useful, please star our repo or consider citing:
```
@inproceedings{deng-etal-2022-prompt,
    title = "Prompt-based Conservation Learning for Multi-hop Question Answering",
    author = "Deng, Zhenyun  and
      Zhu, Yonghua  and
      Chen, Yang  and
      Qi, Qianqian  and
      Witbrock, Michael  and
      Riddle, Patricia",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.154",
    pages = "1791--1800",
}
```
