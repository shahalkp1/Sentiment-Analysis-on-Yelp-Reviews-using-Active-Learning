
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from scipy.stats import entropy
import numpy as np
#query
def multi_argmax(values, n_instances=110):
    max_idx = np.argpartition(-values, n_instances-1, axis=0)[:n_instances]
    return max_idx
def entropy_sampling( X,n_instances= 110):
    ent=np.transpose(entropy(np.transpose(X)))
    return multi_argmax(ent, n_instances=n_instances)
#cleaning
clean_text_pool=clean(rev['text'])
#This will help to create a new csv file in which we will have labelled text and queried text (which need to be labelled)
x_counts = count_vect.transform(clean_text_pool.apply(lambda x: ' '.join(x)))
x_tfidf = transformer.transform(x_counts)
sample=model.predict_proba(x_tfidf)
rev_id=entropy_sampling(sample)
y=[]
train_ai=rev.loc[rev_id]
train_ai.to_csv(r"C:\Users\shaha\Desktop\IBS\yelp\train_data121.csv", mode='a', index=False, header=False)