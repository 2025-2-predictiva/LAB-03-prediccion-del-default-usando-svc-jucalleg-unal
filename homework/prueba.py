import gzip, pickle

# 1) Cargar modelo
with gzip.open("files/models/model.pkl.gz", "rb") as f:
    model = pickle.load(f)

# 2) Cargar los mismos datos que usa el autograder
with open("files/grading/x_train.pkl", "rb") as f: x_train = pickle.load(f)
with open("files/grading/y_train.pkl", "rb") as f: y_train = pickle.load(f)
with open("files/grading/x_test.pkl",  "rb") as f: x_test  = pickle.load(f)
with open("files/grading/y_test.pkl",  "rb") as f: y_test  = pickle.load(f)

# 3) Ver los scores (model.score usa el .score del estimador final)
print("train score:", model.score(x_train, y_train))
print("test  score:", model.score(x_test,  y_test))
