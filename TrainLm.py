from fastai.text import *
from fastai.callbacks import * 

path =Path('/home/ubuntu/fastai2/fastai/courses/dl2/imdb_scripts') #Path to where the repository is
trn_lm = np.load(path / 'eswiki/tmp/trn_ids.npy',allow_pickle=True)
val_lm = np.load(path/ 'eswiki/tmp/val_ids.npy',allow_pickle=True)
itos = pickle.load(open(path / 'eswiki/tmp/itos.pkl', 'rb'))
vocab = Vocab(itos)
data_lm = TextLMDataBunch.from_ids(path,vocab,trn_lm, val_lm) 
torch.cuda.empty_cache()
learn = language_model_learner(data_lm,TransformerXL, drop_mult=0.5)
learn.unfreeze()
learn.fit_one_cycle(4, slice(1e-3), callbacks=[SaveModelCallback(learn)])
learn.load('bestmodel')
learn.save('encM2')
print(learn.validate())
