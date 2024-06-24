"""
Generate intermediate queries given dataset
"""
import pickle
from collections import defaultdict
import argparse
from tqdm import tqdm


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


# no need to convert to text here, leave as int for faster computation?
def create_kg(data_path, id2ent, id2rel):
    kg_edges = defaultdict(lambda: defaultdict(set))
    with open(f'{data_path}/train.txt', 'r') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            kg_edges[id2ent[int(h)]][id2rel[int(r)]].add(id2ent[int(t)])

    return kg_edges


def main(args):
    (train_queries, train_answers, ent2id, id2ent, rel2id, id2rel) = load_data(args.data_path)
    # print(ent2id)
    # print(id2ent)
    print("Loaded data.")


    kg_edges = create_kg(args.data_path, id2ent, id2rel)
    print("Created KG.")
    print(kg_edges)
    return

    intermediate_entities = defaultdict(lambda: defaultdict(set))

    qss = [('e', ('r', 'r')), ('e', ('r', 'r', 'r'))]   # 2p, 3p

    for query_structure in tqdm(qss):
        queries = train_queries[query_structure]

        for query in tqdm(queries):
            start_entity, relations = query

            # entities after first hop
            first_hop_entities = kg_edges[start_entity][relations[0]]

            if query_structure == ('e', ('r', 'r')):
                # 2p
                second_hop_entities = set()
                for e in first_hop_entities:
                    second_hop_entities.update(kg_edges[e][rel2])

                verified_first_hop = set()
                for e in first_hop_entities:
                    if kg_edges[e][rel2] & train_answers[query]:
                        verified_first_hop.add(e)

                intermediate_entities[query] = verified_first_hop

        named_intermediate_entities = {
            (id2ent[q[0]], (id2rel[q[1][0]], id2rel[q[1][1]])): {id2ent[e] for e in ents}
            for q, ents in intermediate_entities.items()
        }

        with open(f'{args.data_path}/train-intermediate-entities.pkl', 'wb') as f:
            pickle.dump(named_intermediate_entities, f)

        return named_intermediate_entities



if __name__=="__main__":
    # train_q = pickle.load(open("./NELL-betae/train-queries.pkl", "rb"))
    # train_a = pickle.load(open("./NELL-betae/train-answers.pkl", "rb"))

    # with open("train_queries_output.txt", "w") as f_q:
    #     f_q.write(str(train_q))

    # with open("train_answers_output.txt", "w") as f_a:
    #     f_a.write(str(train_a))

    # first, handle 2p and 3p queries. 

    print(main(parse_args()))
    