# MPCNN-sentence-similarity
README is on the way. This project is based on paper [**Multi-Perspective Sentence Similarity Modeling
with Convolutional Neural Networks**](http://ttic.uchicago.edu/~kgimpel/papers/he+etal.emnlp15.pdf). 

### Information

1. dataset: SICK.txt
2. pre-trained word vector: gloVe-300-dimension.
3. pearson_train **r**: 0.99
4. pearson_test **r**: 0.80

**Cannot achieve 0.90 as the paper said.**

### To train the model

```bash
$ python train.py
```

### To evaluate the result

```bash
$ python eval.py
```

### To view the result on Tensorboard

```bash
$ tensorboard --logdir=runs/your_model/summaries
```
replace *your_model* with the path to your summary directory.
