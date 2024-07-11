"""
Primary file for running GQE.
"""
import argparse
import os
import pickle
import datetime
from collections import defaultdict, Counter
import torch
from torch.utils.data import DataLoader
from dataloader import TrainDataset, TestDataset
from util import *
from model import GQE
from tqdm import tqdm
import transformers
import ast

from huggingface_hub import login
login("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")

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

query_name_dict = {('e', ('r',)): '1p',
                   ('e', ('r', 'r')): '2p',
                   ('e', ('r', 'r', 'r')): '3p',
                   (('e', ('r',)), ('e', ('r',))): '2i',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                   ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip', # to do 
                   (('e', ('r', 'r')), ('e', ('r',))): 'pi',    # to do
                   (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                   ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp', # to do 
                   (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin', # to do
                   (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni', # to do 
                   (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                   ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                   }
name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(name_query_dict.keys())  # ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2in', '3in', 'inp', 'pin',
# 'pni', '2u-DNF', '2u-DM', 'up-DNF', 'up-DM']

CURR_TIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# "paths" that start at the same entity and have same length
# paths are constructed by finding queries that start at the same entity and
# have the same length, and by getting its answer and appending it to the query
# to get the final path
def get_path_based_context(query, all_queries, answers, k=3):
    paths = []
    for q in all_queries:
        if q[0] == query[0] and len(q) == len(query) and q != query:
            if q in answers:
                full_path = q + (next(iter(answers[q])),)
                paths.append(full_path)
    print("PATHS: ", paths)
    return paths[:k]


# entity neightborhood of the entity w.r.t to first relation in the query
def get_entity_neighborhood(entity, relation, all_queries, k=5):
    neighbors = set()
    for q in all_queries:
        if q[0] == entity and q[1][0] == relation:
            neighbors.add(q[-1])
    return list(neighbors)[:k]


# relations that freq appear together with the query relation in the KG
# more context about the structure and patterns 
def get_relation_co_occurrence(relation, all_queries, k=3):
    co_occurrences = Counter()
    for q in all_queries:
        if relation in q[1:-1]:
            for r in q[1:-1]:
                if r != relation:
                    co_occurrences[r] += 1
    return co_occurrences.most_common(k)


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--do_train', action='store_true', help="do train")
    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")

    parser.add_argument('--load_model', action='store_true', help="load model", default=False)
    parser.add_argument('--load_test_size', default=10, type=int, help="batch size for test set when we load model")
    parser.add_argument('--model_path', type=str, default='', help="model path when we want to load model")

    parser.add_argument('--use_llm', action='store_true', default=False)

    parser.add_argument('-g', '--gamma', default=12.0, type=float, help="margin in the loss")
    parser.add_argument('--data_path', type=str, default=None, help="KG data path", required=True)
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int,
                        help="negative entities sampled per query")
    parser.add_argument('-d', '--hidden_dim', default=500, type=int, help="embedding dimension")
    parser.add_argument('--num_epochs', default=5, type=int, help="number of epochs")
    parser.add_argument('-b', '--batch_size', default=32, type=int, help="batch size of queries")
    parser.add_argument('--test_batch_size', default=1, type=int, help='valid/test batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('--val_every_n_epochs', default=2, type=int, help="number of epochs after which to run "
                                                                          "validation on")

    parser.add_argument('--tasks', default='1p.2p.3p.2i.3i', type=str,
                        help="tasks connected by dot, only supports 1p, 2p, 3p, 2i, and 3i")

    parser.add_argument('-evu', '--evaluate_union', default="DNF", type=str, choices=['DNF', 'DM'],
                        help='the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use '
                             'the De Morgan\'s laws (DM)')

    return parser.parse_args(args)


def load_data(path, tasks, args):
    print("Loading data...")
    train_queries = pickle.load(open(os.path.join(path, "train-queries.pkl"), 'rb'))
    train_answers = pickle.load(open(os.path.join(path, "train-answers.pkl"), 'rb'))
    valid_queries = pickle.load(open(os.path.join(path, "valid-queries.pkl"), 'rb'))
    valid_hard_answers = pickle.load(open(os.path.join(path, "valid-hard-answers.pkl"), 'rb'))
    valid_easy_answers = pickle.load(open(os.path.join(path, "valid-easy-answers.pkl"), 'rb'))
    test_queries = pickle.load(open(os.path.join(path, "test-queries.pkl"), 'rb'))
    test_hard_answers = pickle.load(open(os.path.join(path, "test-hard-answers.pkl"), 'rb'))
    test_easy_answers = pickle.load(open(os.path.join(path, "test-easy-answers.pkl"), 'rb'))

    # remove tasks not in args.tasks
    for name in all_tasks:
        if 'u' in name:
            name, evaluate_union = name.split('-')
        else:
            evaluate_union = args.evaluate_union
        if name not in tasks or evaluate_union != args.evaluate_union:
            query_structure = name_query_dict[name if 'u' not in name else '-'.join([name, evaluate_union])]
            if query_structure in train_queries:
                del train_queries[query_structure]
            if query_structure in valid_queries:
                del valid_queries[query_structure]
            if query_structure in test_queries:
                del test_queries[query_structure]

    with open(os.path.join(path, "stats.txt")) as f:
        lines = f.readlines()
        num_entities = int(lines[0].split(' ')[-1])
        num_relations = int(lines[1].split(' ')[-1])

    return (train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries,
            test_hard_answers, test_easy_answers, num_entities, num_relations)


# needs to be updated to include test results within results
def save_model(model, save_dir, results, hyperparameters, model_name: str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_path = os.path.join(save_dir, f"{CURR_TIME}_{model_name}")

    torch.save(model.state_dict(), model_path)

    results_path = os.path.join(save_dir, f"{CURR_TIME}_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"train_path_metrics: {results[0]}\n")
        f.write(f"train_other_metrics: {results[1]}\n")
        f.write(f"val_metrics: {results[2]}\n")
        f.write(f"val_final_metrics: {results[3]}\n")
        f.write(f"Details:\n")
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")
        f.write("\n *** \n")


def average_logs(logs):
    avg_log = {
        'positive_sample_loss': sum(log['positive_sample_loss'] for log in logs) / len(logs),
        'negative_sample_loss': sum(log['negative_sample_loss'] for log in logs) / len(logs),
        'loss': sum(log['loss'] for log in logs) / len(logs),
    }
    return avg_log


def train(model, optimizer, train_path_dataloader, train_other_dataloader, valid_dataloader, val_answers, args):
    model.train().to(DEVICE)

    train_path_metrics, train_other_metrics = [], []
    val_metrics = None

    if args.do_valid:
        print("Val after {} epochs...".format(args.val_every_n_epochs))

    if train_other_dataloader is not None:
        print("Doing training on both path and other queries per epoch.")

    for i in range(args.num_epochs):
        model.train().to(DEVICE)

        epoch_path, epoch_other = [], []

        print('Epoch {}/{}'.format(i + 1, args.num_epochs))
        print("path_dataloader:")
        for (positives, negatives, flattened_queries, query_structures) in tqdm(train_path_dataloader):
            log = GQE.train_step(model, optimizer, positives, negatives, flattened_queries, query_structures, DEVICE)
            epoch_path.append(log)
            # print(log)

        if train_other_dataloader is not None:
            print("other_dataloader:")
            for (positives, negatives, flattened_queries, query_structures) in tqdm(train_other_dataloader):
                log = GQE.train_step(model, optimizer, positives, negatives, flattened_queries, query_structures,
                                     DEVICE)
                epoch_other.append(log)
                # print(log)

        if epoch_path:
            avg_path_log = average_logs(epoch_path)
            train_path_metrics.append(avg_path_log)
            print(avg_path_log)

        if epoch_other:
            avg_other_log = average_logs(epoch_other)
            train_other_metrics.append(avg_other_log)
            print(avg_other_log)

        if args.do_valid:
            if i % args.val_every_n_epochs == 0:
                # do validation
                print("val_dataloader:")
                val_metrics = GQE.evaluate(model, val_answers, valid_dataloader, DEVICE)
                print(val_metrics)

    return train_path_metrics, train_other_metrics, val_metrics


def main(args):
    tasks = args.tasks.split('.')
    for task in tasks:
        if 'n' in task:
            assert False, "GQE doesn't work with negation."

    args.save_path = os.path.join('runs', CURR_TIME)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    id2ent = pickle.load(open(os.path.join(args.data_path, 'id2ent.pkl'), 'rb'))
    id2rel = pickle.load(open(os.path.join(args.data_path, 'id2rel.pkl'), 'rb'))
    ent2id = pickle.load(open(os.path.join(args.data_path, 'ent2id.pkl'), 'rb'))
    rel2id = pickle.load(open(os.path.join(args.data_path, 'ent2id.pkl'), 'rb'))

    def id_to_text(entity_id):
        return id2ent[entity_id]

    def rel_to_text(relation_id):
        return id2rel[relation_id]

    def text_to_id(entity_text):
        return ent2id[entity_text]

    def text_to_rel(relation_text):
        return rel2id[relation_text]

    (train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries,
     test_hard_answers, test_easy_answers, num_entities, num_relations) = load_data(args.data_path, tasks, args)

    train_path_dataloader, train_other_dataloader, valid_dataloader, test_dataloader = None, None, None, None

    if args.do_train:
        print("Configuring training data...")
        # print("train_queries.length: {}".format(len(train_queries)))
        train_path_queries, train_other_queries = defaultdict(set), defaultdict(set)

        path_list = ['1p', '2p', '3p']
        for qs in train_queries:
            if query_name_dict[qs] in path_list:
                train_path_queries[qs] = train_queries[qs]
            else:
                train_other_queries[qs] = train_queries[qs]

        # print("train_path_queries.length: {}".format(len(train_path_queries)))
        # print("train_other_queries.length: {}".format(len(train_other_queries)))

        train_path_queries = flatten_query(train_path_queries)

        # print("train_path_queries_flattened.length: {}".format(len(train_path_queries)))

        train_path_dataset = TrainDataset(train_path_queries, num_entities, num_relations, args.negative_sample_size,
                                          train_answers)
        train_path_dataloader = DataLoader(train_path_dataset,
                                           collate_fn=TrainDataset.collate_fn,
                                           batch_size=args.batch_size,
                                           shuffle=True)

        if train_other_queries:
            train_other_queries = flatten_query(train_other_queries)
            train_other_datasets = TrainDataset(train_other_queries, num_entities, num_relations,
                                                args.negative_sample_size, train_answers)
            train_other_dataloader = DataLoader(train_other_datasets,
                                                collate_fn=TrainDataset.collate_fn,
                                                batch_size=args.batch_size,
                                                shuffle=True)
        else:
            train_other_dataloader = None

    if args.do_valid:
        valid_queries = flatten_query(valid_queries)
        valid_dataset = TestDataset(valid_queries, num_entities, num_relations)
        valid_dataloader = DataLoader(valid_dataset,
                                      collate_fn=TestDataset.collate_fn,
                                      batch_size=args.test_batch_size)

    if args.do_test:
        test_queries = flatten_query(test_queries)
        test_dataset = TestDataset(test_queries, num_entities, num_relations)
        test_dataloader = DataLoader(test_dataset,
                                     collate_fn=TestDataset.collate_fn,
                                     batch_size=args.test_batch_size)

    model = GQE(
        num_entities=num_entities,
        num_relations=num_relations,
        embed_dim=args.hidden_dim,
        gamma=args.gamma,
        query_name_dict=query_name_dict
    )
    model.to(DEVICE)

    if args.load_model and args.model_path == '':
        print("If you do --load_model, provide a valid --model_path parameter")
        return

    if args.load_model:
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        print("Loaded model from {}".format(args.model_path))

    print("Tasks: ", tasks)

    if args.do_train:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        # print(GQE.train_step(model, optimizer, train_path_dataloader, DEVICE))
        train_path_metrics, train_other_metrics, val_metrics = train(model, optimizer, train_path_dataloader,
                                                                     train_other_dataloader, valid_dataloader,
                                                                     (valid_easy_answers, valid_hard_answers),
                                                                     args)

        final_val_metrics = None
        if args.do_valid:
            model.eval()
            print("val_dataloader:")
            final_val_metrics = GQE.evaluate(model, (valid_easy_answers, valid_hard_answers), valid_dataloader, DEVICE)
            print(final_val_metrics)

        save_model(model, args.save_path, (train_path_metrics, train_other_metrics, val_metrics, final_val_metrics),
                   hyperparameters={
                       "gamma": args.gamma,
                       "negative_sample_size": args.negative_sample_size,
                       "hidden_dim": args.hidden_dim,
                       "num_epochs": args.num_epochs,
                       "batch_size": args.batch_size,
                       "test_batch_size": args.test_batch_size,
                       "lr": args.learning_rate,
                   },
                   model_name='model.pth')

        if args.do_test:
            print("Testing on model just trained...")
            model.eval()
            test_metrics = GQE.evaluate(model, (test_easy_answers, test_hard_answers), test_dataloader, DEVICE)
            print("Test Metrics: ", test_metrics)

    if args.load_model and args.use_llm:
        print("Loaded model and starting testing. Doing reranking with LLM.")
        model.eval()

        test_queries = flatten_query(test_queries)
        test_dataset = TestDataset(test_queries, num_entities, num_relations)
        test_dataloader = DataLoader(test_dataset,
                                     collate_fn=TestDataset.collate_fn,
                                     batch_size=args.load_test_size)

        logs = defaultdict(list)
        skipped = []

        # test_queries consists of (q, qs) pairs
        all_queries = [q for q, _ in test_queries]
        all_structures = [s for _, s in test_queries]

        print(all_queries[:10])

        with torch.no_grad():
            for (negatives, flattened_queries, queries, query_structures) in tqdm(test_dataloader):
                # positives.to(DEVICE)
                negatives.to(DEVICE)

                batch_queries_dict, batch_idx_dict = defaultdict(list), defaultdict(list)

                for i, query in enumerate(flattened_queries):  # group queries with the same structure
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idx_dict[query_structures[i]].append(i)

                for qs in batch_queries_dict:
                    batch_queries_dict[qs] = torch.LongTensor(batch_queries_dict[qs]).to(DEVICE)

                _, negative_logit, idx = model(None, negatives, batch_queries_dict, batch_idx_dict, DEVICE)

                queries = [queries[i] for i in idx]
                query_structures = [query_structures[i] for i in idx]

                sorted_logits = torch.argsort(negative_logit, dim=1, descending=True).to(DEVICE)
                top10_entities = sorted_logits[:, :10].tolist()

                batch_messages = []
                batch_wid = []

                def extract_relations(query):
                    relations = []
                    def recurse(q):
                        if isinstance(q, tuple):
                            for item in q:
                                if isinstance(item, tuple):
                                    recurse(item)
                                elif isinstance(item, int):
                                    relations.append(item)
                    recurse(query)
                    return relations

                def visualize_pbc(pbc, id2ent, id2rel):
                    print("PBC IN FUNC: ", pbc)
                    if not pbc or len(pbc) < 3:
                        return

                    h, rels, t = pbc 
                    # return [
                    #     id2ent[h],
                    #     [id2rel[r] for r in rels],
                    #     id2ent[t]
                    # ]
                    head_text = id2ent.get(h, "Unknown Entity")
                    relations_text = [id2rel.get(r, "Unknown Relation") for r in rels]
                    tail_text = id2ent.get(t, "Unknown Entity")

                    return f"{head_text} -> {' -> '.join(relations_text)} -> {tail_text}"

                # Need to alter below to print more than 1p.2p
                for query, query_structure, top10 in zip(queries, query_structures, top10_entities):
                    entity = query[0]
                    relations = extract_relations(query[1:])
                  
                    entity_text = id2ent[entity]
                    relations_text = [id2rel[rel] for rel in relations]
                    # print(f"Query: {query} \nEntity: {entity_text}, \nRelations: {relations_text}, \n")


                    # node_degree = get_node_degree(entity, all_queries)
                    path_context = get_path_based_context(query, all_queries, test_hard_answers)
                    entity_neighborhood = get_entity_neighborhood(entity, relations[0], all_queries)
                    # relation_co_occurrence = get_relation_co_occurrence(relations[0], all_queries)

                    # pbc = ', '.join([' -> '.join([id2ent[e[0] if isinstance(e, tuple) else e] for e in path]) for path in path_context])
                    pbc = [visualize_pbc(pbc, id2ent, id2rel) for pbc in path_context]
                    en = ', '.join([id2ent[e[0] if isinstance(e, tuple) else e] for e in entity_neighborhood])
                    # rco = ', '.join([f"{id2rel[rel]} ({count})" for rel, count in relation_co_occurrence])

                    # print("Node Degree: ", node_degree)
                    # print("Path Context: ", pbc)
                    # print("Entity Neighborhood: ", en)
                    # print("Relation Co Occurance: ", rco)


                    top10_entities_text = [id2ent[ent] for ent in top10]
                    # print(f"Top 10 entities: {top10_entities_text}")

                    wid = {i: x for i, x in enumerate(top10_entities_text)}

                    # prompt = f"The query is ({entity_text}, {relations_text}, ?), where the first element is the head, and the following elements are relations, and '?' is the tail entity whose candidates we want to rank. Rerank the following top 10 tail entities for '?' based on their relevance and likelihood of being the correct answer: {wid}. Here, the entities are structured and id-value pairs. Return the reranked list of entity ids only (not the values), ensuring that your reply is a permutation of the top 10 tail entities that were provided above."

                    # prompt = f"\
                    # Query:({entity_text}, {relations_text}, ?).\
                    # 1. Entity Neighborhood: {en}\
                    # 2. Path-Based Context: {pbc}\
                    # Rerank the following top 10 tail entities for '?' based on their relevance and likelihood of being the correct answer, considering the additional context provided: {wid}\
                    # Return the reranked list of entity ids only (not the values) of size 10, ensuring that your reply is a permutation of the tail entity ids that were provided above. Do not output anything other than the list.\
                    # "

                    prompt = f"\
                    Query:({entity_text}, {relations_text}, ?).\
                    1. Entity Neighborhood: {en}\ 
                    2. Path-Based Context: {pbc}\ 
                    Rerank these top 10 tail entities: {wid}\
                    Return only the reranked list of entity ids. Do not output anything other than the list.\
                    "

                    print(prompt)

                    # put idea of context and use context into system message. put geenral instructions into the system message
                    # sort messages by length and group similar length messages 

                    message = {
                    "role": "user",
                    "content": prompt
                    }
                    batch_messages.append(message)
                    batch_wid.append(wid)

                # system_message = {"role": "system", "content": "You are an evaluator tasked with re-ranking entities for some logical queries. \
                # Queries are structured as follows, where the first element is the head and the following elements are relations, and '?' is the tail entity whose candidates we want to rank. \
                # The correct answer to each query is found by starting at the head entity and traversing the graph along the edges specified by the relations, in the order they're given. \
                # The '?' entity candidates are structured and id-value pairs. You only respond with the reranked list of entity ids for each query.\
                # The queries input are in batches, and you have to rerank candidates individually for each query."}
                system_message = {"role": "system", "content": "You are an evaluator tasked with reranking entities for queries. \
                Each query is structured with the head entity followed by relations, and '?' is the tail entity. \
                Rerank candidates individually for each query based on the provided context. \
                Return only the reranked list of entity ids."}
            
                terminators = [
                    pipeline.tokenizer.eos_token_id,
                    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                # Generate LLM responses for the batch
                llm_responses = pipeline(
                    [system_message] + batch_messages,
                    eos_token_id=terminators,
                    max_new_tokens=128, # reduced from 1024 to 128, try with larger batch size -> not this, padding max_length
                    do_sample=True,
                    temperature=0.2,
                    top_p=0.1,
                    num_return_sequences=len(batch_messages)
                )

                new_sorted_logits = sorted_logits.clone()

                for i, (response, wid) in enumerate(zip(llm_responses, batch_wid)):
                    try:
                        reranked_entities = ast.literal_eval(response["generated_text"][-1]["content"])
                        print(f"Query {i + 1} - Reranked entities by LLM: {reranked_entities}")

                        # reranked_entities = ["concept_" + wid[x] for x in reranked_entities]
                        reranked_entities = [wid[x] for x in reranked_entities if x in wid]
                        reranked_ids = [text_to_id(ent.strip()) for ent in reranked_entities if ent.strip() in ent2id.keys()]

                        print(f"Query {i + 1} - Reranked entity IDs: {reranked_entities}, {reranked_ids}")

                        # new_sorted_logits[i, :10] = torch.tensor(reranked_ids, device=DEVICE)

                        if len(reranked_ids) == 10:
                            new_sorted_logits[i, :10] = torch.tensor(reranked_ids, device=DEVICE)
                        else:
                            print(f"Warning: Query {i + 1} returned {len(reranked_ids)} entities instead of 10. Skipping reranking for this query.")
                            skipped.append(f"Skipped Query {i + 1} - Reranked entity IDs: {reranked_entities}, {reranked_ids}")

                    except (SyntaxError, ValueError) as e:
                        print(f"Error processing response for query {i + 1}: {e}")
                        print(f"Raw response: {response['generated_text'][-1]['content']}")

                sorted_logits = new_sorted_logits
                ranked_logits = sorted_logits.clone().to(torch.float).to(DEVICE)
                ranked_logits = ranked_logits.scatter_(1,
                                                       sorted_logits,
                                                       torch.arange(model.num_entities).to(torch.float).repeat(
                                                           sorted_logits.size(0), 1).to(DEVICE)).to(DEVICE)

                for idx, (i, query, query_structure) in enumerate(zip(sorted_logits[:, 0], queries, query_structures)):
                    hard_answer, easy_answer = test_hard_answers[query], test_easy_answers[query]
                    num_hard, num_easy = len(hard_answer), len(easy_answer)

                    curr_ranking = ranked_logits[idx, list(easy_answer) + list(hard_answer)].to(DEVICE)
                    curr_ranking, indices = torch.sort(curr_ranking)

                    masks = indices >= num_easy
                    answer_list = torch.arange(num_hard + num_easy).to(torch.float).to(DEVICE)

                    curr_ranking = curr_ranking - answer_list + 1
                    curr_ranking = curr_ranking[masks].to(DEVICE)

                    mrr = torch.mean(1.0 / curr_ranking).item()
                    hit_at_10 = torch.mean((curr_ranking <= 10).to(torch.float)).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS10': hit_at_10,
                        'num_hard_answer': num_hard,
                    })

        metrics = defaultdict(lambda: defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]]) / len(
                    logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        # test_metrics = GQE.evaluate_with_llm(model, (test_easy_answers, test_hard_answers), test_dataloader, DEVICE, pipeline, id2ent, id2rel, text_to_id)
        print("TEST METRICS: Loading in model for testing: ", metrics)
        # print(skipped)

    elif args.load_model:
        model.eval()
        print("WITHOUT RERANKING USING LLM")

        test_queries = flatten_query(test_queries)
        test_dataset = TestDataset(test_queries, num_entities, num_relations)
        test_dataloader = DataLoader(test_dataset,
                                     collate_fn=TestDataset.collate_fn,
                                     batch_size=10)

        test_metrics = GQE.evaluate(model, (test_easy_answers, test_hard_answers), test_dataloader, DEVICE)
        print("Test Metrics: ", test_metrics)



if __name__ == '__main__':
    print("Running at {} on {}".format(CURR_TIME, "CUDA" if torch.cuda.is_available() else "CPU"))
    main(parse_args())

    # entid = pickle.load(open("./NELL-betae/ent2id.pkl", 'rb'))
    # print(entid.keys())

    # messages = [
    # {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    # {"role": "user", "content": "Who are you?"},
    # ]

    # terminators = [
    #     pipeline.tokenizer.eos_token_id,
    #     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    # ]

    # outputs = pipeline(
    #     messages,
    #     max_new_tokens=256,
    #     eos_token_id=terminators,
    #     do_sample=True,
    #     temperature=0.6,
    #     top_p=0.9,
    # )
    # print(outputs[0]["generated_text"][-1])

    # tq = pickle.load(open("FB15k-237/test-queries.pkl", "rb"))
    # print(tq)
