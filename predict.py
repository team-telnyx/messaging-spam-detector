import asyncio
import pickle
from typing import Optional, List

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import time

from interface.interfaces import SpamResponse
start = time.time()

class PredictMsgStatus:
    def __init__(
        self,
        vect: CountVectorizer(),
        model: MultinomialNB(), # Naive Bayes Multinomial
    ):

        self.vect = vect
        self.model = model

        resources_folder = 'resources'
        self.msg_train_data = f'{resources_folder}/msg_train_data.pickle'
        self.msg_test_data = f'{resources_folder}/msg_test_data.pickle'
        self.status_train_data = f'{resources_folder}/status_train_data.pickle'
        self.status_test_data = f'{resources_folder}/status_test_data.pickle'

        self.train_files = [self.msg_train_data, self.status_train_data]
    async def load_msg_data(self):
        output_list = []
        for file in self.train_files:
            filehandler = open(file, 'rb')
            filehandler = pickle.load(filehandler)
            output_list.append(filehandler)

        msg_data, status_data = output_list
        msg_array, status_array = np.array(msg_data), np.array(status_data)
        print(len(msg_array),(len(status_array)))
        return msg_array, status_array

    async def predict(self, msg_body: List[str]):
        msg_array, status_array = await self.load_msg_data()
        predict_status_list = []
        error = ''
        msg_vect = self.vect.fit_transform(msg_array)
        self.model.fit(msg_vect, status_array)

        try:
            if len(msg_body) > 0:
                for msg in msg_body:
                    vectorize_message = self.vect.transform([msg])
                    predict = self.model.predict(vectorize_message)
                    if predict == 0:
                        predict_status_list.append(True)
                    else:
                        predict_status_list.append(False)

        except BaseException as ex:
            error = str(type(ex).__name__) + ' ' + str(ex)

        result = SpamResponse(
            spam=predict_status_list,
            error=error
        )
        print(result)
        print(time.time() - start)
        return result

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    messages = []
    client = PredictMsgStatus(model=MultinomialNB(), vect=CountVectorizer())


    loop.run_until_complete(client.predict(msg_body=messages))
