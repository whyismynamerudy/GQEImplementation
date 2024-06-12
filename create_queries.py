import argparse
import pickle
import os
import random
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy

MAX_ANS_NUM = 1e6


def index_dataset(name, force=False):
    base_path = "data/{}".format(name)
    files = ["train.txt", "valid.txt", "test.txt"]
    indexed_files = ["train_indexified.txt", "valid_indexified.txt", "test_indexified.txt"]
    return_flag = True

    for i in range(len(indexed_files)):
        if not os.path.exists(os.path.join(base_path, indexed_files[i])):
            return_flag = False
            break

    if return_flag and not force:
        print("Indexed files already exist. Skipping.")
        return

    ent2id, rel2id, id2rel, id2ent = {}, {}, {}, {}
    entid, relid = 0, 0

    for p, index_p in zip(files, indexed_files):
        fw = open(os.path.join(base_path, index_p), "w")
        with open(os.path.join(base_path, p), "r") as f:
            for line in tqdm(f):
                e1, rel, e2 = line.split('\t')
                e1, rel, e2 = e1.strip(), rel.strip(), e2.strip()

                rel_reverse = '-' + rel
                rel = '+' + rel

                if p == "train.txt":  # include reverse rel for more training triples
                    if e1 not in ent2id.keys():
                        ent2id[e1] = entid
                        id2ent[entid] = e1
                        entid += 1

                    if e2 not in ent2id.keys():
                        ent2id[e2] = entid
                        id2ent[entid] = e2
                        entid += 1

                    if rel not in rel2id.keys():
                        rel2id[rel] = relid
                        id2rel[relid] = rel
                        assert relid % 2 == 0
                        relid += 1

                    if rel_reverse not in rel2id.keys():
                        rel2id[rel_reverse] = relid
                        id2rel[relid] = rel_reverse
                        assert relid % 2 == 1
                        relid += 1

                if e1 in ent2id.keys() and e2 in ent2id.keys():
                    fw.write("\t".join([str(ent2id[e1]), str(rel2id[rel]), str(ent2id[e2])]) + "\n")
                    fw.write("\t".join([str(ent2id[e2]), str(rel2id[rel_reverse]), str(ent2id[e1])]) + "\n")

        fw.close()

    with open(os.path.join(base_path, "stats.txt"), "w") as fw:
        fw.write("numentity: " + str(len(ent2id)) + "\n")
        fw.write("numrelations: " + str(len(rel2id)))
    with open(os.path.join(base_path, 'ent2id.pkl'), 'wb') as handle:
        pickle.dump(ent2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(base_path, 'rel2id.pkl'), 'wb') as handle:
        pickle.dump(rel2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(base_path, 'id2ent.pkl'), 'wb') as handle:
        pickle.dump(id2ent, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(base_path, 'id2rel.pkl'), 'wb') as handle:
        pickle.dump(id2rel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Indexing finished.")


def construct_graph(base_path, indexified_files):
    ent_in, ent_out = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))
    for index_p in tqdm(indexified_files):
        with open(os.path.join(base_path, index_p)) as f:
            for line in f:
                if len(line) == 0:
                    continue
                e1, rel, e2 = line.split('\t')
                e1 = int(e1.strip())
                e2 = int(e2.strip())
                rel = int(rel.strip())
                ent_out[e1][rel].add(e2)
                ent_in[e2][rel].add(e1)

    return ent_in, ent_out


def list2tuple(l):
    return tuple(list2tuple(x) if type(x) == list else x for x in l)


def tuple2list(t):
    return list(tuple2list(x) if type(x) == tuple else x for x in t)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='FB15k-237')
    parser.add_argument('--gen_id', type=int, required=True)
    parser.add_argument('--reindex', default=False, action='store_true')
    parser.add_argument('--save_name', default=False, action='store_true')

    args = parser.parse_args()
    return args


def convert_to_textual_query(query, id2ent, id2rel):
    # print("Query: {}".format(query))
    if isinstance(query, tuple):
        ent = query[0]
        if isinstance(ent, int):
            ent = id2ent[ent]
        else:
            ent = convert_to_textual_query(ent, id2ent, id2rel)
        rels = tuple(id2rel[r] if r >= 0 else 'neg-' + id2rel[abs(r) - 1] for r in query[1])
        return ent, rels
    else:
        raise ValueError("Unexpected query format")


# only to be called on 1p queries
def write_links(dataset, ent_out, small_ent_out, name, id2ent, id2rel):
    queries = defaultdict(set)
    tp_answers = defaultdict(set)
    fn_answers = defaultdict(set)
    fp_answers = defaultdict(set)
    num_more_answer = 0

    for ent in ent_out:
        for rel in ent_out[ent]:
            if len(ent_out[ent][rel]) <= MAX_ANS_NUM:
                queries[('e', ('r',))].add((ent, (rel,)))
                tp_answers[(ent, (rel,))] = small_ent_out[ent][rel]
                fn_answers[(ent, (rel,))] = ent_out[ent][rel]
            else:
                num_more_answer += 1

    textual_queries = {q: {convert_to_textual_query(v, id2ent, id2rel) for v in vals} for q, vals in queries.items()}
    textual_tp_answers = {convert_to_textual_query(q, id2ent, id2rel): {id2ent[a] for a in ans} for q, ans in
                          tp_answers.items()}
    textual_fn_answers = {convert_to_textual_query(q, id2ent, id2rel): {id2ent[a] for a in ans} for q, ans in
                          fn_answers.items()}
    textual_fp_answers = {convert_to_textual_query(q, id2ent, id2rel): {id2ent[a] for a in ans} for q, ans in
                          fp_answers.items()}

    with open('./data/%s/%s-queries.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(textual_queries, f)
    with open('./data/%s/%s-tp-answers.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(textual_tp_answers, f)
    with open('./data/%s/%s-fn-answers.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(textual_fn_answers, f)
    with open('./data/%s/%s-fp-answers.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(textual_fp_answers, f)
    print(num_more_answer)

    # with open('./data/%s/%s-queries.pkl' % (dataset, name), 'wb') as f:
    #     pickle.dump(queries, f)
    # with open('./data/%s/%s-tp-answers.pkl' % (dataset, name), 'wb') as f:
    #     pickle.dump(tp_answers, f)
    # with open('./data/%s/%s-fn-answers.pkl' % (dataset, name), 'wb') as f:
    #     pickle.dump(fn_answers, f)
    # with open('./data/%s/%s-fp-answers.pkl' % (dataset, name), 'wb') as f:
    #     pickle.dump(fp_answers, f)
    # print(num_more_answer)


def ground_queries(dataset, query_structure, ent_in, ent_out, small_ent_in, small_ent_out, gen_num, query_name, mode,
                   ent2id, rel2id, id2ent, id2rel):
    num_sampled, num_try, num_repeat, num_more_answer, num_broken, num_no_extra_answer, num_no_extra_negative, num_empty = 0, 0, 0, 0, 0, 0, 0, 0
    tp_ans_num, fp_ans_num, fn_ans_num = [], [], []
    queries = defaultdict(set)
    tp_answers = defaultdict(set)
    fp_answers = defaultdict(set)
    fn_answers = defaultdict(set)

    while num_sampled < gen_num:
        num_try += 1
        empty_query_structure = deepcopy(query_structure)
        answer = random.sample(ent_in.keys(), 1)[0]

        broken_flag = fill_query(empty_query_structure, ent_in, ent_out, answer, ent2id, rel2id)
        if broken_flag:
            num_broken += 1
            continue

        query = empty_query_structure
        answer_set = achieve_answer(query, ent_in, ent_out)
        small_answer_set = achieve_answer(query, small_ent_in, small_ent_out)

        if len(answer_set) == 0:
            num_empty += 1
            continue
        if mode != 'train':
            if len(answer_set - small_answer_set) == 0:
                num_no_extra_answer += 1
                continue
            if 'n' in query_name:
                if len(small_answer_set - answer_set) == 0:
                    num_no_extra_negative += 1
                    continue
        if max(len(answer_set - small_answer_set), len(small_answer_set - answer_set)) > MAX_ANS_NUM:
            num_more_answer += 1
            continue
        if list2tuple(query) in queries[list2tuple(query_structure)]:
            num_repeat += 1
            continue
        queries[list2tuple(query_structure)].add(list2tuple(query))
        tp_answers[list2tuple(query)] = small_answer_set
        fp_answers[list2tuple(query)] = small_answer_set - answer_set
        fn_answers[list2tuple(query)] = answer_set - small_answer_set
        num_sampled += 1
        tp_ans_num.append(len(tp_answers[list2tuple(query)]))
        fp_ans_num.append(len(fp_answers[list2tuple(query)]))
        fn_ans_num.append(len(fn_answers[list2tuple(query)]))

    name_to_save = "{}-{}".format(mode, query_name)

    textual_queries = {q: {convert_to_textual_query(v, id2ent, id2rel) for v in vals} for q, vals in queries.items()}
    textual_tp_answers = {convert_to_textual_query(q, id2ent, id2rel): {id2ent[a] for a in ans} for q, ans in
                          tp_answers.items()}
    textual_fn_answers = {convert_to_textual_query(q, id2ent, id2rel): {id2ent[a] for a in ans} for q, ans in
                          fn_answers.items()}
    textual_fp_answers = {convert_to_textual_query(q, id2ent, id2rel): {id2ent[a] for a in ans} for q, ans in
                          fp_answers.items()}

    with open("./data/{}/{}-queries.pkl".format(dataset, name_to_save), 'wb') as f:
        pickle.dump(textual_queries, f)
    with open("./data/{}/{}-fp-answers.pkl".format(dataset, name_to_save), 'wb') as f:
        pickle.dump(textual_fp_answers, f)
    with open("./data/{}/{}-fn-answers.pkl".format(dataset, name_to_save), 'wb') as f:
        pickle.dump(textual_fn_answers, f)
    with open("./data/{}/{}-tp-answers.pkl".format(dataset, name_to_save), 'wb') as f:
        pickle.dump(textual_tp_answers, f)
    return queries, tp_answers, fp_answers, fn_answers

    # with open("./data/{}/{}-queries.pkl".format(dataset, name_to_save), 'wb') as f:
    #     pickle.dump(queries, f)
    # with open("./data/{}/{}-fp-answers.pkl".format(dataset, name_to_save), 'wb') as f:
    #     pickle.dump(fp_answers, f)
    # with open("./data/{}/{}-fn-answers.pkl".format(dataset, name_to_save), 'wb') as f:
    #     pickle.dump(fn_answers, f)
    # with open("./data/{}/{}-tp-answers.pkl".format(dataset, name_to_save), 'wb') as f:
    #     pickle.dump(tp_answers, f)
    # return queries, tp_answers, fp_answers, fn_answers


def generate_queries(dataset, query_structures, gen_num, query_names, save_name):
    base_path = "./data/{}".format(dataset)
    indexified_files = ['train_indexified.txt', 'valid_indexified.txt', 'test_indexified.txt']

    train_ent_in, train_ent_out = construct_graph(base_path, indexified_files[:1])
    valid_ent_in, valid_ent_out = construct_graph(base_path, indexified_files[:2])
    valid_only_ent_in, valid_only_ent_out = construct_graph(base_path, indexified_files[1:2])
    test_ent_in, test_ent_out = construct_graph(base_path, indexified_files[:3])
    test_only_ent_in, test_only_ent_out = construct_graph(base_path, indexified_files[2:3])

    ent2id = pickle.load(open(os.path.join(base_path, "ent2id.pkl"), 'rb'))
    rel2id = pickle.load(open(os.path.join(base_path, "rel2id.pkl"), 'rb'))
    id2ent = pickle.load(open(os.path.join(base_path, "id2ent.pkl"), 'rb'))
    id2rel = pickle.load(open(os.path.join(base_path, "id2rel.pkl"), 'rb'))

    assert len(query_structures) == 1
    idx = 0
    query_structure = query_structures[idx]
    query_name = query_names[idx] if save_name else str(idx)
    print("General structure: {}, Name: {}".format(query_structure, query_name))

    if query_structure == ['e', ['r']]:
        write_links(dataset, train_ent_out, defaultdict(lambda: defaultdict(set)), 'train-' + query_name, id2ent, id2rel)
        write_links(dataset, valid_only_ent_out, train_ent_out, 'valid-' + query_name, id2ent, id2rel)
        write_links(dataset, test_only_ent_out, valid_ent_out, 'test-' + query_name, id2ent, id2rel)
        print("Link prediction created.")
        return

    # name_to_save = query_name
    # num_sampled, num_try, num_repeat, num_more_answer, num_broken, num_empty = 0, 0, 0, 0, 0, 0
    # train_ans_num = []

    train_queries, train_tp_answers, train_fp_answers, train_fn_answers = ground_queries(dataset, query_structure,
                                                                                         train_ent_in, train_ent_out,
                                                                                         defaultdict(
                                                                                             lambda: defaultdict(set)),
                                                                                         defaultdict(
                                                                                             lambda: defaultdict(set)),
                                                                                         gen_num[0],
                                                                                         query_name, 'train', ent2id,
                                                                                         rel2id, id2ent, id2rel)

    valid_queries, valid_tp_answers, valid_fp_answers, valid_fn_answers = ground_queries(dataset, query_structure,
                                                                                         valid_ent_in, valid_ent_out,
                                                                                         train_ent_in, train_ent_out,
                                                                                         gen_num[1],
                                                                                         query_name, 'valid', ent2id,
                                                                                         rel2id, id2ent, id2rel)

    test_queries, test_tp_answers, test_fp_answers, test_fn_answers = ground_queries(dataset, query_structure,
                                                                                     test_ent_in, test_ent_out,
                                                                                     valid_ent_in, valid_ent_out,
                                                                                     gen_num[2],
                                                                                     query_name, 'test', ent2id, rel2id, id2ent, id2rel)


def fill_query(query_structure, ent_in, ent_out, answer, ent2id, rel2id):
    all_relation_flag = True

    for ele in query_structure[-1]:
        if ele not in ['r', 'n']:
            all_relation_flag = False
            break

    if all_relation_flag:
        r = -1
        for i in range(len(query_structure[-1]))[::-1]:
            if query_structure[-1][i] == 'n':
                query_structure[-1][i] = -2
                continue

            found = False
            for j in range(40):
                r_tmp = random.sample(ent_in[answer].keys(), 1)[0]
                if r_tmp // 2 != r // 2 or r_tmp == r:
                    r = r_tmp
                    found = True
                    break

            if not found:
                return True

            query_structure[-1][i] = r
            answer = random.sample(ent_in[answer][r], 1)[0]

        if query_structure[0] == 'e':
            query_structure[0] = answer
        else:
            return fill_query(query_structure[0], ent_in, ent_out, answer, ent2id, rel2id)

    else:
        same_structure = defaultdict(list)

        for i in range(len(query_structure)):
            same_structure[list2tuple(query_structure[i])].append(i)

        for i in range(len(query_structure)):
            if len(query_structure[i]) == 1 and query_structure[i][0] == 'u':
                assert i == len(query_structure) - 1
                query_structure[i][0] = -1
                continue

            broken_flag = fill_query(query_structure[i], ent_in, ent_out, answer, ent2id, rel2id)

            if broken_flag:
                return True

        for structure in same_structure:
            if len(same_structure[structure]) != 1:
                structure_set = set()

                for i in same_structure[structure]:
                    structure_set.add(list2tuple(query_structure[i]))

                if len(structure_set) < len(same_structure[structure]):
                    return True


def achieve_answer(query, ent_in, ent_out):
    all_relation_flag = True
    for ele in query[-1]:
        if (type(ele) != int) or (ele == -1):
            all_relation_flag = False
            break

    if all_relation_flag:
        if type(query[0]) == int:
            ent_set = set([query[0]])
        else:
            ent_set = achieve_answer(query[0], ent_in, ent_out)

        for i in range(len(query[-1])):
            if query[-1][i] == -2:
                ent_set = set(range(len(ent_in))) - ent_set
            else:
                ent_set_traverse = set()
                for ent in ent_set:
                    ent_set_traverse = ent_set_traverse.union(ent_out[ent][query[-1][i]])
                ent_set = ent_set_traverse

    else:
        ent_set = achieve_answer(query[0], ent_in, ent_out)
        union_flag = False

        if len(query[-1]) == 1 and query[-1][0] == -1:
            union_flag = True

        for i in range(1, len(query)):
            if not union_flag:
                ent_set = ent_set.intersection(achieve_answer(query[i], ent_in, ent_out))
            else:
                if i == len(query) - 1:
                    continue
                ent_set = ent_set.union(achieve_answer(query[i], ent_in, ent_out))

    return ent_set


def main(dataset, gen_id, reindex=False, save_name=False):
    index_dataset(dataset, reindex)

    # Need to include data for the WNRR18 dataset
    train_num_dict = {'FB15k': 273710, "FB15k-237": 149689, "NELL": 107982, 'WN18RR-text': 80000}
    valid_num_dict = {'FB15k': 8000, "FB15k-237": 5000, "NELL": 4000, 'WN18RR-text': 2000}
    test_num_dict = {'FB15k': 8000, "FB15k-237": 5000, "NELL": 4000, 'WN18RR-text': 2000}

    gen_train_num = train_num_dict[dataset]
    gen_valid_num = valid_num_dict[dataset]
    gen_test_num = test_num_dict[dataset]

    e = 'e'
    r = 'r'
    n = 'n'
    u = 'u'
    query_structures = [
        [e, [r]],
        [e, [r, r]],
        [e, [r, r, r]],
        [[e, [r]], [e, [r]]],
        [[e, [r]], [e, [r]], [e, [r]]],
        [[e, [r, r]], [e, [r]]],
        [[[e, [r]], [e, [r]]], [r]],
        # negation
        [[e, [r]], [e, [r, n]]],
        [[e, [r]], [e, [r]], [e, [r, n]]],
        [[e, [r, r]], [e, [r, n]]],
        [[e, [r, r, n]], [e, [r]]],
        [[[e, [r]], [e, [r, n]]], [r]],
        # union
        [[e, [r]], [e, [r]], [u]],
        [[[e, [r]], [e, [r]], [u]], [r]]
    ]
    query_names = ['1p', '2p', '3p', '2i', '3i', 'pi', 'ip', '2in', '3in', 'pin', 'pni', 'inp', '2u', 'up']
    generate_queries(dataset, query_structures[gen_id:gen_id + 1], [gen_train_num, gen_valid_num, gen_test_num],
                     query_names[gen_id:gen_id + 1], save_name)


if __name__ == '__main__':
    # args = parse_args()
    # main(args.dataset, args.gen_id, args.reindex, args.save_name)

    # test0querues = pickle.load(open('./data/WN18RR-text/test-0-tp-answers.pkl', 'rb'))
    # print(test0querues)

    fbtwothreeseven = pickle.load(open("./FB15k-237/ent2id.pkl", 'rb'))
    print(fbtwothreeseven)
