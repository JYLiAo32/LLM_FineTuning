#!/bin/bash

# 收集在完整数据集上的数据
# # 32
# python src/evaluation.py --answer_path results/251212_172211/answer.json

# # 64
# python src/evaluation.py --answer_path results/251216_145427/answer.json

# # 128
# python src/evaluation.py --answer_path results/251216_161739/answer.json

# # 16
# python src/evaluation.py --answer_path results/251216_174056/answer.json

# 评估在验证集上的数据
# 32
python src/evaluation.py --answer_path results/251212_172211/answer_val.json

# 64
python src/evaluation.py --answer_path results/251216_145427/answer_val.json

# 128
python src/evaluation.py --answer_path results/251216_161739/answer_val.json

# 16
python src/evaluation.py --answer_path results/251216_174056/answer_val.json
