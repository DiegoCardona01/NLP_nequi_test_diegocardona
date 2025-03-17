import pandas as pd
import joblib

df = pd.read_csv('../dataset/dataset_predict/predictions_data.csv')
df.head()

texts = df['narrative'].tolist()
print(texts)

map_predictions = {
    '0': 'Credit Reporting',
    '1': 'Debt Collection',
    '2': 'Loans',
    '3': 'Credit Card Services',
    '4': 'Bank Accounts and Services'
}

clf = joblib.load('../model/v_2025-03-14/model_2025-03-14.pkl')

tfidf = joblib.load('../model/vector_2025-03-14/vectorizer_2025-03-14.pkl')

X_test = tfidf.transform(texts)
predict_value = (clf.predict_proba(X_test)[0]).tolist()

pred = (clf.predict(X_test)).tolist()

pred = [map_predictions[str(i)] for i in pred]

pred = list(zip(texts, pred, predict_value))

pred = pd.DataFrame(pred, columns=['narrative', 'prediction', 'predict_value'])

pred.head()
