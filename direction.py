import pandas as pd
import numpy as np
#df=pd.read_csv('telegram_data_with_btc_prices.csv')
df=pd.read_csv('reddit_discussions_with_btc_prices.csv')
df['next_price']=df['btc_closing_price'].shift(-1)
direction =[0]



df.drop(df.tail(1).index,inplace=True)
for row in range(1,len(df['next_price'])):
    if df['next_price'].iloc[row]>df['next_price'].iloc[row-1]:
        direction.append(1)
    else:
        direction.append(0)
df['direction']=direction

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer, accuracy_score
import xgboost as xgb
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
feats = vectorizer.fit_transform(df['text'].values.astype('U'))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feats, df['direction'], test_size=0.33, random_state=42)

# Initialize logistic regression with custom class weights
model = xgb.XGBClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print( f1_score(y_test, y_pred))
 
 
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd

 


np.unique(direction,return_counts=True)

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import f1_score, make_scorer, accuracy_score
        from xgboost import XGBClassifier
        vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        feats = vectorizer.fit_transform(df['text'].values.astype('U'))
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(feats, df['direction'], test_size=0.33, random_state=42)


y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

# Adjust the threshold
threshold = 0.75
pred = (y_pred_proba >= threshold).astype(int)


pos=[]
neg=[]
neu=[]
comp=[]
sentences=list(df['text'])

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
         for sentence in sentences:
             vs = analyzer.polarity_scores(sentence)
             print(str(vs))
             pos.append(vs['pos'])
             neg.append(vs['neg'])
             neu.append(vs['neu'])
             comp.append(vs['compound'])
train_df['neg']=neg
train_df['neu']=neu
train_df['pos']=pos 
train_df['compound']=comp

target_names = ['class 0', 'class 1']
print(classification_report(y_test, pred, target_names=target_names))