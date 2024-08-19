import string
import nltk     #thư viện dành cho NLP
# nltk.download('stopwords')
# nltk.download('punkt')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("2cls_spam_text_cls.csv")

mess = df["Message"].values.tolist()
label = df["Category"].values.tolist()

def preprocess_text(text): #Tiền xử lí các text
    Input_text = text
    # Cho tất cả về kí tự thường
    Input_text = Input_text.lower()
    # Bỏ các dấu '
    Input_text = Input_text.translate(str.maketrans("", "",string.punctuation))
    # Split các từ trong string thành mảng
    Input_text = nltk.word_tokenize(Input_text)
    # Bỏ các từ không ảnh hưởng đến mô hình
    stop_words = nltk.corpus.stopwords.words('english')
    Input_text = [token for token in Input_text if token not in stop_words]
    # Biến các từ trong mảng thành từ "chuẩn"
    stemmer = nltk.PorterStemmer()
    Input_text = [stemmer.stem(token) for token in Input_text]

    return Input_text

mess = [preprocess_text(message) for message in mess]  #Tạo list từ với tất cả câu trong df

def create_dic(mess):
    dic = []
    for tokens in mess:
        for token in tokens:
            if token not in dic:
                dic.append(token)
    return dic

dic = create_dic(mess)  #Tạo danh sách các từ trong df

def create_feature(tokens,dic):
    features = np.zeros(len(dic))
    for token in tokens:
        if token in dic:
            features[dic.index(token)] += 1
    return features

x = np.array([create_feature(tokens,dic) for tokens in mess])

le = LabelEncoder()
y = le.fit_transform(label)

VAL_SIZE = 0.2
TEST_SIZE = 0.125
SEED = 0

X_train, X_val, y_train, y_val = train_test_split(x, y,
                                                  test_size=VAL_SIZE,
                                                  shuffle=True,
                                                  random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                    test_size=TEST_SIZE,
                                                    shuffle=True,
                                                    random_state=SEED)
model = GaussianNB()
print('Start training...')
model = model.fit(X_train, y_train)
print('Training completed!')

y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Val accuracy: {val_accuracy}')
print(f'Test accuracy: {test_accuracy}')

def predict(text, model, dictionary):
    processed_text = preprocess_text(text)
    features = create_feature(text, dictionary)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    prediction_cls = le.inverse_transform(prediction)[0]
    return prediction_cls

test_input = 'I am actually thinking a way of doing something useful'
prediction_cls = predict(test_input, model, dic)
print(f'Prediction: {prediction_cls}')
