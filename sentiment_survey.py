from textblob import TextBlob
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
from nltk.stem.porter import PorterStemmer
nltk.download('wordnet')
import seaborn as sns
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_excel("Mock_Results_Text.xlsx")

df_new = df[['Location','Business Unit','Answer']]

df_new.dropna(subset=['Answer'],inplace=True)


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

#corpus = df_new['Answer_processed'].to_list()

#textblob
Final = pd.DataFrame()   
for i in range(0, df_new.shape[0]):
    
    blob = TextBlob(df_new.iloc[i,2])
    
    temp = pd.DataFrame({'Answer': df_new.iloc[i,3], 'Polarity': blob.sentiment.polarity}, index = [0])
    Final = Final.append(temp)

#vader
FinalResults_Vader = pd.DataFrame()
analyzer = SentimentIntensityAnalyzer()

for i in range(0, df_new.shape[0]):
    
    snt = analyzer.polarity_scores(df_new.iloc[i,3])
    
    temp1 = pd.DataFrame({'Answers': df_new.iloc[i,2], 'Polarity': list(snt.items())[3][1]}, index = [0])

    FinalResults_Vader = FinalResults_Vader.append(temp1)


df_new['Polarity'] = Final['Polarity'].values

def plot(graph):
    graph.set_xticklabels(graph.get_xticklabels(),rotation=90,ha='right')
    plt.show


loc_boxplot=sns.boxplot(x='Location',y='Polarity',data=df_new)
plot(loc_boxplot)

bu_boxplot=sns.boxplot(x='Business Unit',y='Polarity',data=df_new)
plot(bu_boxplot)








