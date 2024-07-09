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

    print("Loaded data.")

    kg_edges = create_kg(args.data_path)
    print("Created KG.")

    intermediate_entities = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    qss = [
        ('e', ('r', 'r')),  # 2p
        ('e', ('r', 'r', 'r')),  # 3p
        ((('e', ('r',)), ('e', ('r',))), ('r',)),  # ip, 0 instances in NELL
        (('e', ('r', 'r')), ('e', ('r',))),  # pi, 0 instances in NELL
        ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)),  # inp
        (('e', ('r', 'r')), ('e', ('r', 'n'))),  # pin
        (('e', ('r', 'r', 'n')), ('e', ('r',)))  # pni
    ]

    for query_structure in qss:
        queries = train_queries[query_structure]
        print(len(queries))

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

                first_hop_entities = kg_edges[start_entity][relations[0]]  

                for answer in train_answers[query]:
                    for e1 in first_hop_entities:
                        second_hop_entities = kg_edges[e1][relations[1]]

                        for e2 in second_hop_entities:
                            if answer in kg_edges[e2][relations[2]]:
                                intermediate_entities[query][answer]['1p'].add(e1)
                                intermediate_entities[query][answer]['2p'].add(e2)


            elif query_structure == ((('e', ('r',)), ('e', ('r',))), ('r',)):
                # ip

                ((start_entity1, relation1), (start_entity2, relation2)), final_relation = query

                for answer in train_answers[query]:
                    entities1 = kg_edges[start_entity1][relation1[0]]
                    entities2 = kg_edges[start_entity2][relation2[0]]

                    intersection = entities1.intersection(entities2)
                    for e in intersection:
                        if answer in kg_edges[e][final_relation[0]]:
                            intermediate_entities[query][answer]['ip'].add(e)


                # intermediate between 2p v
            elif query_structure == (('e', ('r', 'r')), ('e', ('r',))):
                # pi - pretty much the answer?

                (start_entity1, relations1), (start_entity2, relation2) = query

                for answer in train_answers[query]:
                    entities1 = set()

                    for e1 in kg_edges[start_entity1][relations1[0]]:
                        entities1.update(kg_edges[e1][relations1[1]])

                    entities2 = kg_edges[start_entity2][relation2[0]]
                    intersection = entities1.intersection(entities2)
                    intermediate_entities[query][answer]['pi'] = intersection


            elif query_structure == ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)):
                # inp

                ((start_entity1, relation1), (start_entity2, (relation2, negation))), final_relation = query

                for answer in train_answers[query]:
                    entities1 = kg_edges[start_entity1][relation1[0]]
                    entities2 = set()

                    for e in kg_edges[start_entity2][relation2]:
                        if negation not in kg_edges[e]:
                            entities2.add(e)

                    intersection = entities1.intersection(entities2)
                    for e in intersection:
                        if answer in kg_edges[e][final_relation[0]]:
                            intermediate_entities[query][answer]['inp'].add(e)

            
            elif query_structure == (('e', ('r', 'r')), ('e', ('r', 'n'))):
                # pin - same as pi, pretty much answer

                (start_entity1, relations1), (start_entity2, (relation2, negation)) = query

                for answer in train_answers[query]:

                    entities1 = set()
                    for e1 in kg_edges[start_entity1][relations1[0]]:
                        entities1.update(kg_edges[e1][relations1[1]])

                    entities2 = set()
                    for e in kg_edges[start_entity2][relation2]:
                        if negation not in kg_edges[e]:
                            entities2.add(e)

                    intersection = entities1.intersection(entities2)
                    intermediate_entities[query][answer]['pin'] = intersection


            elif query_structure == (('e', ('r', 'r', 'n')), ('e', ('r',))):
                # pni

                (start_entity1, (relation1, relation2, negation)), (start_entity2, relation3) = query

                for answer in train_answers[query]:

                    entities1 = set()

                    for e1 in kg_edges[start_entity1][relation1]:
                        for e2 in kg_edges[e1][relation2]:
                            if negation not in kg_edges[e2]:
                                entities1.add(e2)

                    entities2 = kg_edges[start_entity2][relation3[0]]
                    intersection = entities1.intersection(entities2)
                    intermediate_entities[query][answer]['pni'] = intersection


    # intermediate_entities_dict = {}
    # for q, ans_dict in intermediate_entities.items():
    #     intermediate_entities_dict[q] = {}
    #     for ans, hop_dict in ans_dict.items():
    #         intermediate_entities_dict[q][ans] = {hop: set(ents) for hop, ents in hop_dict.items()}

    # with open(f'{args.data_path}/train-intermediate-entities-id.pkl', 'wb') as f:
    #     pickle.dump(intermediate_entities_dict, f)

    # named_intermediate_entities = {}
    # for q, ans_dict in intermediate_entities.items():
    #     named_q = (id2ent[q[0]], tuple(id2rel[r] for r in q[1]))
    #     named_ans_dict = {}
    #     for ans, hop_dict in ans_dict.items():
    #         named_ans = id2ent[ans]
    #         named_hop_dict = {
    #             hop: {id2ent[e] for e in ents}
    #             for hop, ents in hop_dict.items()
    #         }
    #         named_ans_dict[named_ans] = named_hop_dict
    #     named_intermediate_entities[named_q] = named_ans_dict

    # with open(f'{args.data_path}/train-intermediate-entities-named.pkl', 'wb') as f:
    #     pickle.dump(named_intermediate_entities, f)

    def convert_query_to_tuple(q):
        if isinstance(q, tuple):
            return tuple(convert_query_to_tuple(item) for item in q)
        elif isinstance(q, list):
            return tuple(convert_query_to_tuple(item) for item in q)
        else:
            return q

    def convert_to_named(item, id2ent, id2rel):
        if isinstance(item, tuple):
            return tuple(convert_to_named(i, id2ent, id2rel) for i in item)
        elif isinstance(item, list):
            return [convert_to_named(i, id2ent, id2rel) for i in item]
        elif isinstance(item, str) and item in {'r', 'n'}:
            return item
        elif item in id2ent:
            return id2ent[item]
        elif item in id2rel:
            return id2rel[item]
        else:
            return item

    # convert to dictionary and save as ID version
    print("Converting to dictionary and save as ID version")
    intermediate_entities_dict = {}
    for q, ans_dict in tqdm(intermediate_entities.items()):
        q_tuple = convert_query_to_tuple(q)
        intermediate_entities_dict[q_tuple] = {}
        for ans, hop_dict in ans_dict.items():
            intermediate_entities_dict[q_tuple][ans] = {hop: set(ents) for hop, ents in hop_dict.items()}

    with open(f'{args.data_path}/train-intermediate-entities-id.pkl', 'wb') as f:
        pickle.dump(intermediate_entities_dict, f)

    # convert to named entities and relations
    print("Converting to named entities and relations")
    named_intermediate_entities = {}
    for q, ans_dict in tqdm(intermediate_entities.items()):
        named_q = convert_to_named(q, id2ent, id2rel)
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


    # for each answer there are different intermediate entities
    # store intermediate entity *for every answer*
    # as a set, can't differentiate between which entity leads to which answer

    # query : {answer: intermediate}



if __name__=="__main__":
    # train_q = pickle.load(open("./NELL-betae/train-queries.pkl", "rb"))

    # with open("train_answers_output.txt", "w") as f_a:
    #     f_a.write(str(train_a))

    # first, handle 2p and 3p queries. 

    # main(parse_args())

    # named_intermediate_entities = pickle.load(open(f"./FB15k-betae/train-intermediate-entities-named.pkl", "rb"))
    # count = 0
    # for k in named_intermediate_entities:
    #     print(k, named_intermediate_entities[k])
    #     count += 1
    #     if count == 10:
    #         break

    print("15K")
    intermediate_entities_dict = pickle.load(open(f"./FB15k-betae/train-intermediate-entities-id.pkl", "rb"))
    count = 0
    for k in intermediate_entities_dict:
        print(k, intermediate_entities_dict[k])
        count += 1
        if count == 5:
            break

    print("15K 237")
    intermediate_entities_dict = pickle.load(open(f"./FB15k-237/train-intermediate-entities-id.pkl", "rb"))
    count = 0
    for k in intermediate_entities_dict:
        print(k, intermediate_entities_dict[k])
        count += 1
        if count == 5:
            break

    print("NELL")
    intermediate_entities_dict = pickle.load(open(f"./NELL-betae/train-intermediate-entities-id.pkl", "rb"))
    count = 0
    for k in intermediate_entities_dict:
        print(k, intermediate_entities_dict[k])
        count += 1
        if count == 5:
            break
        