"""
Generate intermediate queries given dataset
"""
import pickle

if __name__=="__main__":
    train_q = pickle.load(open("./NELL-betae/train-queries.pkl", "rb"))
    train_a = pickle.load(open("./NELL-betae/train-answers.pkl", "rb"))

    with open("train_queries_output.txt", "w") as f_q:
        f_q.write(str(train_q))

    with open("train_answers_output.txt", "w") as f_a:
        f_a.write(str(train_a))