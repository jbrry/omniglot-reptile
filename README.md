# omniglot-reptile

REPTILE algorithm: https://arxiv.org/abs/1803.02999

This will mainly be a copy of the TensorFlow version here (https://github.com/openai/supervised-reptile), re-implemented for self-study.

# Setup
```
conda create -n omniglot-reptile python=3.7
conda activate omniglot-reptile

pip install -r requirements.txt
```

# Fetch data
```
./scripts/fetch_omniglot_data.sh
```

# Reproducing training runs

```
# transductive 1-shot 5-way Omniglot.
python -u reptile/main.py --shots 1 --inner-batch 10 --inner-iters 5 --meta-step 1 --meta-batch 5 --meta-iters 100000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_o15t --transductive

# 5-shot 5-way Omniglot.
python -u reptile/main.py --train-shots 10 --inner-batch 10 --inner-iters 5 --learning-rate 0.001 --meta-step 1 --meta-step-final 0 --meta-batch 5 --meta-iters 100000 --eval-batch 5 --eval-iters 50 --checkpoint ckpt_o55

# 1-shot 5-way Omniglot.
python -u reptile/main.py --shots 1 --inner-batch 10 --inner-iters 5 --meta-step 1 --meta-batch 5 --meta-iters 100000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_o15

# 1-shot 20-way Omniglot.
python -u reptile/main.py --shots 1 --classes 20 --inner-batch 20 --inner-iters 10 --meta-step 1 --meta-batch 5 --meta-iters 200000 --eval-batch 10 --eval-iters 50 --learning-rate 0.0005 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_o120

# 5-shot 20-way Omniglot.
python -u reptile/main.py --classes 20 --inner-batch 20 --inner-iters 10 --meta-step 1 --meta-batch 5 --meta-iters 200000 --eval-batch 10 --eval-iters 50 --learning-rate 0.0005 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_o520
```

