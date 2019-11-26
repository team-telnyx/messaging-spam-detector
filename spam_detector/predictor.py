import pickle
from typing import NamedTuple, List, Optional

import numpy as np
import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


class SpamResponse(NamedTuple):
    is_spam: bool
    error: Optional[str]


class MessageStatusPredictor:
    def __init__(self):

        self.vect = CountVectorizer()
        self.model = MultinomialNB()
        self.load_data()

    def load_data(self):

        resources_folder = 'resources'
        self.combined_sms_mix = f'{resources_folder}/combined_sms_mix.csv'
        data = pandas.read_csv('combined_sms_mix.csv', encoding='latin-1')
        train_data = data[:]
        vectorize_text = self.vect.fit_transform(train_data.message_body)
        self.model.fit(vectorize_text, train_data.status)

    def predict(self, msg_body: str) -> SpamResponse:
        error = None
        is_spam = None

        try:
            if len(msg_body) > 0:
                vectorize_message = self.vect.transform([msg_body])
                predict = self.model.predict(vectorize_message)
                if predict == 'failed':
                    is_spam = True
                else:
                    is_spam = False

        except BaseException as ex:
            error = str(type(ex).__name__) + ' ' + str(ex)

        result = SpamResponse(
            is_spam=is_spam,
            error=error
        )
        return result
