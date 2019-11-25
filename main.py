import asyncio
from typing import Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.multiclass import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import *
import pandas
from interface.interfaces import SpamResponse
class SpamFilter(object):
    def __init__(
        self,
        user_id: str,
        msg_body: Optional[List[str]],
        data: str = 'combined_sms_mix.csv'
    ):
        self.user_id = user_id
        self.msg_body = msg_body
        self.data = data

        Vectorizer = CountVectorizer()
        Classifier = MultinomialNB()
        self.Vectorizer = Vectorizer
        self.Classifier = Classifier

    def load_trained_data(self):
        data = pandas.read_csv('combined_sms_mix.csv', encoding='latin-1')
        train_data = data[:]
        vectorize_text = self.Vectorizer.fit_transform(train_data.message_body)
        self.Classifier.fit(vectorize_text, train_data.status)

    async def filter_spam(self, msg_body: Optional[List[str]] = None) -> SpamResponse:
        self.load_trained_data()
        predict_probability = []
        error = ''
        predict_spam_list = []

        try:
            if len(msg_body) > 0:
                for msg in msg_body:
                    vectorize_message = self.Vectorizer.transform([msg])
                    predict = self.Classifier.predict(vectorize_message)[0]
                    if predict == 'failed':
                        predict_spam_list.append(True)
                    else:
                        predict_spam_list.append(False)
                    # predict_probability = Classifier.predict_proba(vectorize_message).tolist()

        except BaseException as ex:
            error = str(type(ex).__name__) + ' ' + str(ex)

        result = SpamResponse(
            spam=predict_spam_list,
            error=error
        )
        return result


if __name__ == '__main__':

    messages = ['We got something nice for you']
    spam_class = SpamFilter(
        user_id='1',
        msg_body=messages
    )
    loop = asyncio.get_event_loop()
    loop.run_until_complete(spam_class.filter_spam(msg_body=messages))