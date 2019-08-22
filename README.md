# SpanishTransformerXL
Language model trained in wiki corpus with fastai v1 acc>43% len(vocab)=60K perplexity 18,9

fastai v1.0.57
spacy


# Preparing wiki corpus

run prepare_wiki.sh:
bash prepare_wiki.sh "es"

wrap wiki file:
python wiki2csv.py

download spacy for spa:
python -m spacy download es_core_news_sm

create tokens and map tokens to Ids (save itos):
python create_toks_1Ids.py --dir-path eswiki --lang es_core_news_sm

run notebook or TrainLm.py (I trained 5 epochs with different learning rates )

to train a classifier with the pretrained model weights:
run notebook LM_fine_tune_Classifier_training.ipynb

# Pretrained Model and itos:

itos:
https://1drv.ms/u/s!AjpGgbeu726PhB-XVsfvRzJAUZH-?e=6RvMP2

Model:
https://1drv.ms/u/s!AjpGgbeu726PhCBWMpzi7GfHmmcz?e=2TfVUR
