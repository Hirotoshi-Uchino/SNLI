from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence


class SNLI:

    def __init__(self, train_df, dev_df, test_df, maxlen):
        self.maxlen  = maxlen
        self.x_train = None
        self.x_dev   = None
        self.x_test  = None
        self.y_train = None
        self.y_dev   = None
        self.y_test  = None
        self.tokenizer = None
        self.preprocessing(train_df, dev_df, test_df)


    def preprocessing(self, train_df, dev_df, test_df):
        train_df = train_df[train_df["gold_label"] != "-"].fillna("")
        dev_df = dev_df[dev_df["gold_label"] != "-"].fillna("")
        test_df = test_df[test_df["gold_label"] != "-"].fillna("")

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_df["sentence1"])
        tokenizer.fit_on_texts(train_df["sentence2"])
        tokenizer.fit_on_texts(dev_df["sentence1"])
        tokenizer.fit_on_texts(dev_df["sentence2"])
        tokenizer.fit_on_texts(test_df["sentence1"])
        tokenizer.fit_on_texts(test_df["sentence2"])

        seq_train1 = tokenizer.texts_to_sequences(train_df["sentence1"])
        seq_train2 = tokenizer.texts_to_sequences(train_df["sentence2"])
        seq_dev1   = tokenizer.texts_to_sequences(dev_df["sentence1"])
        seq_dev2   = tokenizer.texts_to_sequences(dev_df["sentence2"])
        seq_test1  = tokenizer.texts_to_sequences(test_df["sentence1"])
        seq_test2  = tokenizer.texts_to_sequences(test_df["sentence2"])

        x_train1 = sequence.pad_sequences(seq_train1, maxlen=self.maxlen)
        x_train2 = sequence.pad_sequences(seq_train2, maxlen=self.maxlen)
        self.x_train = [x_train1, x_train2]

        y_label = {"contradiction": 0, "entailment": 1, "neutral": 2}
        y_train = [y_label[i] for i in train_df["gold_label"]]
        self.y_train = np_utils.to_categorical(y_train, 3)

        x_dev1 = sequence.pad_sequences(seq_dev1, maxlen=self.maxlen)
        x_dev2 = sequence.pad_sequences(seq_dev2, maxlen=self.maxlen)
        self.x_dev = [x_dev1, x_dev2]

        y_dev = [y_label[i] for i in dev_df["gold_label"]]
        self.y_dev = np_utils.to_categorical(y_dev, 3)

        x_test1 = sequence.pad_sequences(seq_test1, maxlen=self.maxlen)
        x_test2 = sequence.pad_sequences(seq_test2, maxlen=self.maxlen)
        self.x_test = [x_test1, x_test2]

        y_test = [y_label[i] for i in test_df["gold_label"]]
        self.y_test = np_utils.to_categorical(y_test, 3)

        self.tokenizer = tokenizer