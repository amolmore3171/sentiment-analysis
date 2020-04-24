
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
from nltk.stem.porter import PorterStemmer
nltk.download('wordnet')
import seaborn as sns
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_excel("Comments_4_22_Survey.xlsx")

df_new = df[['Location','Business Unit','Question','Answer']]

df_new=df_new.dropna()
df_new=df_new.reset_index(drop=True)


spec_chars = ["!",'"',"#","%","&","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–"]

#text preprocess
stop = stopwords.words('english')  
df_new['Answer_processed'] = df_new['Answer'].apply(lambda j: ' '.join([item for item in j.split() if item not in stop]))

for char in spec_chars:
    df_new['Answer_processed'] = df_new['Answer_processed'].str.replace(char, '').str.lower()
    
stemmer = PorterStemmer()
df_new['Answer_processed'] = df_new['Answer_processed'].apply(lambda j: ' '.join([stemmer.stem(item) for item in j.split()]))

length=df_new[df_new['Answer_processed'].map(len)<2].index
df_new.drop(length,inplace=True)
df_new=df_new.reset_index(drop=True)


#vader
FinalResults_Vader = pd.DataFrame()
analyzer = SentimentIntensityAnalyzer()

df_new['scores'] = df_new['Answer'].apply(lambda ans: analyzer.polarity_scores(ans))

df_new['compound'] = df_new['scores'].apply(lambda score_dict: score_dict['compound'])
df_new['positive'] = df_new['scores'].apply(lambda score_dict: score_dict['pos'])
df_new['negative'] = df_new['scores'].apply(lambda score_dict: score_dict['neg'])
df_new['neutral'] = df_new['scores'].apply(lambda score_dict: score_dict['neu'])

df_new['pred_sentiment'] = df_new['compound'].apply(lambda c: 'Positive' if c>0 else 'Negative')

df_new.to_csv('sentiment_predicted.csv',index=False)

'''
def plot(graph):
    graph.set_xticklabels(graph.get_xticklabels(),rotation=90,ha='right')
    graph.set_ylabel("Sentiment")
    plt.show()


loc_boxplot=sns.boxplot(x='Location',y='compound',data=df_new)
plot(loc_boxplot)

bu_boxplot=sns.boxplot(x='Business Unit',y='compound',data=df_new)
plot(bu_boxplot)
'''









