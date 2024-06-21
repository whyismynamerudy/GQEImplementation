"""
Generate intermediate queries given dataset
"""
import pickle
from collections import defaultdict
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, required=True, help="path to dataset")

    args = parser.parse_args()
    return args


def load_data(data_path):
    with open(f'{data_path}/train-queries.pkl', 'rb') as f:
        train_queries = pickle.load(f)
    with open(f'{data_path}/train-answers.pkl', 'rb') as f:
        train_answers = pickle.load(f)
    with open(f'{data_path}/ent2id.pkl', 'rb') as f:
        ent2id = pickle.load(f)
    with open(f'{data_path}/id2ent.pkl', 'rb') as f:
        id2ent = pickle.load(f)
    with open(f'{data_path}/rel2id.pkl', 'rb') as f:
        rel2id = pickle.load(f)
    with open(f'{data_path}/id2rel.pkl', 'rb') as f:
        id2rel = pickle.load(f)
    
    return (train_queries, train_answers, ent2id, id2ent, rel2id, id2rel)


def create_kg(data_path):
    kg_edges = defaultdict(lambda: defaultdict(set))
    with open(f'{data_path}/train.txt', 'r') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            kg_edges[ent2id[h]][rel2id[r]].add(ent2id[t])

    return kg_edges


def main(args):
    data = load_data(args.data_path)
    kg = create_kg(args.data_path)


if __name__=="__main__":
    # train_q = pickle.load(open("./NELL-betae/train-queries.pkl", "rb"))
    # train_a = pickle.load(open("./NELL-betae/train-answers.pkl", "rb"))

    # with open("train_queries_output.txt", "w") as f_q:
    #     f_q.write(str(train_q))

    # with open("train_answers_output.txt", "w") as f_a:
    #     f_a.write(str(train_a))

    main(parse_args())
    