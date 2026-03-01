import pickle

log_reg = pickle.load(open('log_reg.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_fitted.pkl', 'rb'))

data = "this movie was great this was good"
new_data = tfidf.transform([data])
output = log_reg.predict(new_data)
print(output[0])
