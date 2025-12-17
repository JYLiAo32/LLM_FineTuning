#!/bin/bash
# 收集各模型的答案
# 32
python src/generation.py --exp_name 251212_172211 --checkpoint None
python src/generation.py --exp_name 251212_172211 --checkpoint 200
python src/generation.py --exp_name 251212_172211 --checkpoint 550

# 64
python src/generation.py --exp_name 251216_145427 --checkpoint 200
python src/generation.py --exp_name 251216_145427 --checkpoint 550

# 128
python src/generation.py --exp_name 251216_161739 --checkpoint 200
python src/generation.py --exp_name 251216_161739 --checkpoint 550

# 16
python src/generation.py --exp_name 251216_174056 --checkpoint 200
python src/generation.py --exp_name 251216_174056 --checkpoint 550
