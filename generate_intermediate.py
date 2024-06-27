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
def create_kg(data_path):
    kg_edges = defaultdict(lambda: defaultdict(set))
    with open(f'{data_path}/train.txt', 'r') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            kg_edges[int(h)][int(r)].add(int(t))

    return kg_edges


def main(args):
    (train_queries, train_answers, ent2id, id2ent, rel2id, id2rel) = load_data(args.data_path)
    # print(ent2id)
    # print(id2ent)
    print("Loaded data.")


    kg_edges = create_kg(args.data_path)
    print("Created KG.")
    # print(kg_edges)
    # with open("kg_edges.txt", "w") as f_a:
    #     f_a.write(str(kg_edges))

    intermediate_entities = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    qss = [('e', ('r', 'r')), ('e', ('r', 'r', 'r'))]   # 2p, 3p

    for query_structure in qss:
        queries = train_queries[query_structure]

        num_printed=0

        print("QS: {}".format(query_structure))
        for query in tqdm(queries):
            start_entity, relations = query # decompose query

            if query_structure == ('e', ('r', 'r')):
                # 2p

                # entities after first hop
                first_hop_entities = kg_edges[start_entity][relations[0]]

                for answer in train_answers[query]:
                    # verified_first_hop = set()

                    # iterate thru all the potential one-hop entities
                    for e in first_hop_entities:

                        # if current answer entitiy appears in answer of 2p query, add it
                        if answer in kg_edges[e][relations[1]]:
                            intermediate_entities[query][answer]['1p'].add(e)

                    # intermediate_entities[query][answer]['1p'] = verified_first_hop
                    # if len(verified_first_hop) == 1 and num_printed < 5:
                    #     print("1 intermediate only :( query {} with answer {}".format(query, answer))
                    #     num_printed += 1

            elif query_structure == ('e', ('r', 'r', 'r')):
                # 3p

                # entities after first hop
                first_hop_entities = kg_edges[start_entity][relations[0]]  

                for answer in train_answers[query]:
                    for e1 in first_hop_entities:
                        second_hop_entities = kg_edges[e1][relations[1]]
                        for e2 in second_hop_entities:
                            if answer in kg_edges[e2][relations[2]]:
                                intermediate_entities[query][answer]['1p'].add(e1)
                                intermediate_entities[query][answer]['2p'].add(e2)


    intermediate_entities_dict = {}
    for q, ans_dict in intermediate_entities.items():
        intermediate_entities_dict[q] = {}
        for ans, hop_dict in ans_dict.items():
            intermediate_entities_dict[q][ans] = {hop: set(ents) for hop, ents in hop_dict.items()}

    with open(f'{args.data_path}/train-intermediate-entities-id.pkl', 'wb') as f:
        pickle.dump(intermediate_entities_dict, f)

    named_intermediate_entities = {}
    for q, ans_dict in intermediate_entities.items():
        named_q = (id2ent[q[0]], tuple(id2rel[r] for r in q[1]))
        named_ans_dict = {}
        for ans, hop_dict in ans_dict.items():
            named_ans = id2ent[ans]
            named_hop_dict = {
                hop: {id2ent[e] for e in ents}
                for hop, ents in hop_dict.items()
            }
            named_ans_dict[named_ans] = named_hop_dict
        named_intermediate_entities[named_q] = named_ans_dict

    with open(f'{args.data_path}/train-intermediate-entities-named.pkl', 'wb') as f:
        pickle.dump(named_intermediate_entities, f)

    count = 0
    for k in named_intermediate_entities:
        print(named_intermediate_entities[k])
        count += 1
        if count == 5:
            break

    count = 0
    for k in intermediate_entities_dict:
        print(intermediate_entities_dict[k])
        count += 1
        if count == 5:
            break

    # for each answer there are different intermediate entities
    # store intermediate entity *for every answer*
    # as a set, can't differentiate between which entity leads to which answer

    # query : {answer: intermediate}



if __name__=="__main__":
    # train_q = pickle.load(open("./NELL-betae/train-queries.pkl", "rb"))

    # with open("train_answers_output.txt", "w") as f_a:
    #     f_a.write(str(train_a))

    # first, handle 2p and 3p queries. 

    main(parse_args())

    train_i = pickle.load(open("./NELL-betae/train-intermediate-entities-named.pkl", "rb"))
    with open("train_queries_intermediate-named.txt", "w") as f_q:
        f_q.write(str(train_i))

    train_i = pickle.load(open("./NELL-betae/train-intermediate-entities-id.pkl", "rb"))
    with open("train_queries_intermediate-id.txt", "w") as f_q:
        f_q.write(str(train_i))
    