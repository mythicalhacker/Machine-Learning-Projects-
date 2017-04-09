import pandas as pd

pd.set_option('display.mpl_style','default')
figsize(15,5)

#Reading datasets according to fields and separators
df=pd.read_csv("./smsspamcollection/SMSSpamCollection",sep='\t',header=None,names=['label','sms_message'])

#df.head() - to verify the imported data

#Data Preprocessing according to string ham or spam

df['label']=df.label.map({'ham':0,'spam':1})

#df.shape() - to get the number of rows and colums



#IMPLEMENTING BAG OF WORDS ON A SAMPLE DATA WITHOUT scikit-learn

#Step 1 : convert into lower case 

'''
documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

lower_case_documents = []
for i in documents:
    lower_case_documents.append(i.lower())
print(lower_case_documents)'''

#step 2:  remove all punctuation 

'''sans_punctuation_documents = []
import string

for i in lower_case_documents:
    sans_punctuation_documents.append(i.translate(str.maketrans('', '', string.punctuation)))
print(sans_punctuation_documents)

'''

#step 3: Tokenization


'''
preprocessed_documents = []
for i in sans_punctuation_documents:
    preprocessed_documents.append(i.split(' '))
print(preprocessed_documents)
'''

#step 4: count frequencies

'''
frequency_list = []
import pprint
from collections import Counter

for i in preprocessed_documents:
    frequency_counts = Counter(i)
    frequency_list.append(frequency_counts)
pprint.pprint(frequency_list)

'''



#IMPLEMENTTING BAG OF WORDS FOR sms_message colums data with  scikit-learn

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                 
                                                    df['label'], 
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))



from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()

training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)



#NAIVE BAYE'S IMPLEMENTATIOJN

from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)
predictions = naive_bayes.predict(testing_data)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))