import asyncio
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class CreateData(object):
    def __init__(self, input_data_file: str, test_size = .01):
        self.input_data_file = input_data_file
        self.test_size = test_size

        resources_folder = 'resources'
        self.msg_train_data = f'{resources_folder}/msg_train_data.pickle'
        self.msg_test_data = f'{resources_folder}/msg_test_data.pickle'
        self.status_train_data = f'{resources_folder}/status_train_data.pickle'
        self.status_test_data = f'{resources_folder}/status_test_data.pickle'

    def load_raw_data(self):
        df = pd.read_csv(self.input_data_file, dtype={'status': str})
        df["label_status"] = df.status.map({"delivered": 0, "failed": 1})

        msg_body = df["message_body"]
        labels = df["label_status"]
        return msg_body, labels

    async def train_spam_detector(self):

        msg_body, labels = self.load_raw_data()
        (
            msg_train_data,
            msg_test_data,
            status_train_data,
            status_test_data,
        ) = train_test_split(msg_body, labels, test_size=self.test_size)
        msg_train_data = np.array(msg_train_data)
        msg_test_data = np.array(msg_test_data)
        status_train_data = np.array(status_train_data)
        status_test_data = np.array(status_test_data)

        msg_train_output = open(self.msg_train_data, "wb")
        pickle.dump(msg_train_data, msg_train_output)
        msg_train_output.close()

        msg_test_output = open(self.msg_test_data, "wb")
        pickle.dump(msg_test_data, msg_test_output)
        msg_test_output.close()

        status_train_output = open(self.status_train_data, "wb")
        pickle.dump(status_train_data, status_train_output)
        status_train_output.close()

        status_test_output = open(self.status_test_data, "wb")
        pickle.dump(status_test_data, status_test_output)
        status_test_output.close()


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    client = CreateData(input_data_file='')
    loop.run_until_complete(client.train_spam_detector())
