"""
Primary file for running GQE.
"""
import argparse
import os
import pickle
import datetime
import random
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from dataloader import TrainDataset, TestDataset
from util import *
from model import GQE
from tqdm import tqdm

query_name_dict = {('e', ('r',)): '1p',
                   ('e', ('r', 'r')): '2p',
                   ('e', ('r', 'r', 'r')): '3p',
                   (('e', ('r',)), ('e', ('r',))): '2i',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                   ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                   (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                   (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                   ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                   (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                   (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
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


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--do_train', action='store_true', help="do train")
    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")

    parser.add_argument('--load_model', action='store_true', help="load model", default=False)
    parser.add_argument('--model_path', type=str, default='', help="model path when we want to load model")

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

    ent2id = pickle.load(open(os.path.join(args.data_path, 'ent2id.pkl'), 'rb'))
    id2ent = pickle.load(open(os.path.join(args.data_path, 'id2ent.pkl'), 'rb'))
    id2rel = pickle.load(open(os.path.join(args.data_path, 'id2rel.pkl'), 'rb'))
    rel2id = pickle.load(open(os.path.join(args.data_path, 'rel2id.pkl'), 'rb'))

    def id_to_text(entity_id):
        return id2ent[entity_id]

    def rel_to_text(relation_id):
        return id2rel[relation_id]

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
        model.load_state_dict(torch.load(args.model_path))
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

    if args.load_model and args.do_test:
        print("Loaded model and starting testing.")
        model.eval()

        sample_queries = random.sample(list(test_queries.keys()), 10)
        for query in sample_queries:
            query_structure = query_name_dict[query]
            flattened_query = flatten_query({query: test_queries[query]})
            test_dataset = TestDataset(flattened_query, num_entities, num_relations)
            test_dataloader = DataLoader(test_dataset,
                                         collate_fn=TestDataset.collate_fn,
                                         batch_size=1)

            for (positives, negatives, queries, query_structures) in test_dataloader:
                print(f"Query: {queries}")
                for entity, relations in queries:
                    entity_text = id_to_text(entity.item())
                    relations_text = [rel_to_text(rel.item()) for rel in relations]
                    print(f"Textual Query: Entity: {entity_text}, Relations: {relations_text}")

                scores = model(queries, query_structures)
                top10_entities = torch.topk(scores, 10)[1].tolist()
                top10_entities_text = [id_to_text(ent) for ent in top10_entities]
                print(f"Top 10 entities: {top10_entities_text}")


if __name__ == '__main__':
    print("Running at {} on {}".format(CURR_TIME, "CUDA" if torch.cuda.is_available() else "CPU"))
    main(parse_args())

    # tq = pickle.load(open("FB15k-237/test-queries.pkl", "rb"))
    # print(list(tq.keys()))
