# CALYPSO: A Neural Network Model for Natural Language Inference


## Authors:
Colin Man
colinman(at)stanford(dot)edu

Kenny Xu
kenxu95(at)stanford(dot)edu

Kat Gregory
katg(at)stanford(dot)edu


## Abstract:
The ability to infer meaning from text has long been regarded as one of the ``benchmarks'' of the quest to artificially approximate human intelligence. The field of Natural Language Inference explores this task by explicitly modeling inference relationships in natural language. In this work, we present the CALYPSO model, which builds upon Chen et al. '16's EBIM model by enhancing the matching layer with modifications to Chen's soft attention as well as three matching algorithms inspired by Wang et al. '17. Although CALYPSO's 82\% accuracy is 3.2\% lower than that of our EBIM implementation, our ablation study and comparison of training loss over time suggest that every modification has value and that hyperparameter tuning as well as revisions to the merging framework promise better results.

Please see our paper and poster for additional information.


## To Run:
python code/main.py --help
for instructions

### Examples:
Create an empty folder called train_params in home directory before running command.

#### EBIM Model
```
python -u code/main.py --dev --batch_size=32 --num_train=-1 --num_dev=-1 --bucket --stmt_processor=bilstm --attentive_matching --lr=.0004 --dropout=0.5  > >(tee train-attention-run.out) 2> >(tee train-attention-run.err >&2)
```

#### CALYPSO Model
```
python -u code/main.py --dev --batch_size=32 --num_train=-1 --num_dev=-1 --bucket --stmt_processor=bilstm --attentive_matching --weight_attention --full_matching --max_attentive_matching --maxpool_matching --lr=.0004 --dropout=0.5  > >(tee train-attention-run.out) 2> >(tee train-attention-run.err >&2)
```

