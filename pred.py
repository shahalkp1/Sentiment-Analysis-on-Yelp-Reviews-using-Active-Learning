#cleaning the text
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.metrics import accuracy_score
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
import nltk
from nltk.corpus import stopwords
my_stopwords = set(stopwords.words('english') + ['super', 'duper', 'place'])
from nltk.tokenize import word_tokenize
import string
def clean(tex):
  tex = tex.replace("&amp;amp;", '').replace("\'",'') 
  tex=tex.str.replace(r'\d+','')
  tex = tex.str.replace('@user', '')
  tex = tex.str.replace("[^a-zA-Z#]", " " )
  tex = tex.str.replace("#", " ")
  tex = tex.str.replace("-", " ")
  tex = tex.str.replace("$", " ")
  tex = tex.apply(lambda x: x.split())
  #tex= tex.apply(lambda x: [stemmer.stem(i) for i in x])
  def process(text):
    # Check characters to see if they are in punctuation
    nopunc = set(char for char in list(text) if char not in string.punctuation)
    # Join the characters to form the string.
    nopunc = " ".join(nopunc)
    # remove any stopwords if present
    return [word for word in nopunc.lower().split() if word.lower() not in my_stopwords]
  #tex = tex.apply(process)
  return tex


pool_data= pd.read_csv(r"**data",index_col=False)
train_data_text=clean(pool_data['text'])
count_vect = CountVectorizer(stop_words='english')
transformer = TfidfTransformer(norm='l2',sublinear_tf=True)
x_train_counts = count_vect.fit_transform(train_data_text.apply(lambda x: ' '.join(x)))
x_train_tfidf = transformer.fit_transform(x_train_counts)

import joblib
model = joblib.load(open('mod.pkl','rb'))

text='I go to blow bar to get my brows done by natalie (brow specialist) which i highly recommend she is great does a great job on my eyebrows! But then i got a blow by victoria!! Wow i was impress i have thin, straight, dead hair and she left me with the biggest volume ive ever had!!! Tried another girl but didnt like it as much so victoria will be my girl for ever; very beautiful clean place!!!'

#cleaning the text for testing
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
import nltk
from nltk.corpus import stopwords
my_stopwords = set(stopwords.words('english') + ['super', 'duper', 'place'])
from nltk.tokenize import word_tokenize
import string
def clean(tex):
  tex = tex.replace("&amp;amp;", '').replace("\'",'') 
  tex=tex.replace(r'\d+','')
  tex = tex.replace('@user', '')
  tex = tex.replace("[^a-zA-Z#]", " " )
  tex = tex.replace("#", " ")
  tex = tex.replace('(','').replace(')','')  
  tex = tex.replace("!", " ")
  tex = tex.replace("-", " ")
  tex = tex.replace("$", " ")
  tex = tex.replace(",", " ")
  tex = tex.replace(";", " ")
  tex = tex.split()  
  tex=[str(i) for i in tex]
  nopunc = set(char for char in tex if char not in string.punctuation)
    # Join the characters to form the string.
  nopunc = " ".join(nopunc)
  return [word for word in nopunc.lower().split() if word.lower() not in my_stopwords]


train_data_text=clean(text)
train_data_text=' '.join(train_data_text)
train_data_text=pd.Series(train_data_text)
x_test_counts = count_vect.transform(train_data_text)
x_tst_tfidf = transformer.transform(x_test_counts)

model.predict(x_tst_tfidf)
