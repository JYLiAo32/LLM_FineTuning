#!/bin/bash
# 分割各模型的答案，以便单独评估在验证集上的数据
# 32
python src/split_answer.py --answer_path results/251212_172211/answer.json

# 64
python src/split_answer.py --answer_path results/251216_145427/answer.json

# 128
python src/split_answer.py --answer_path results/251216_161739/answer.json

# 16
python src/split_answer.py --answer_path results/251216_174056/answer.json
