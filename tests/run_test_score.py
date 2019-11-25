from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
import pandas
import csv

data = pandas.read_csv('combined_sms_mix.csv', encoding='latin-1')
test_data_csv = pandas.read_csv('test_spam.csv', encoding='latin-1')
train_data = data[:]
test_data = test_data_csv[:]

classifier = OneVsRestClassifier(SVC(kernel='linear'))
vectorizer = TfidfVectorizer()

# train_module
vectorize_text = vectorizer.fit_transform(train_data.message_body)
classifier.fit(vectorize_text, train_data.status)

# score
vectorize_text = vectorizer.transform(test_data.message_body)
score = classifier.score(vectorize_text, test_data.status)
print(score) # 98,8


csv_arr = []
for index, row in test_data.iterrows():
    answer = row[2]
    text = row[3]
    vectorize_text = vectorizer.transform([text])
    predict = classifier.predict(vectorize_text)[0]
    if predict == answer:
        result = 'right'
    else:
        result = 'wrong'
    csv_arr.append([len(csv_arr), text, answer, predict, result])


# write csv
with open('test_score_spam.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['#', 'text', 'answer', 'predict', result])

    for row in csv_arr:
        spamwriter.writerow(row)
