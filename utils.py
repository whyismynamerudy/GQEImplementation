import argparse
import pickle


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, required=True, help="path to dataset")

    args = parser.parse_args()
    return args


def load_data(data_path):
    with open(f'{data_path}/test-easy-answers.pkl', 'rb') as f:
        easy_answers = pickle.load(f)
    with open(f'{data_path}/test-hard-answers.pkl', 'rb') as f:
        hard_answers = pickle.load(f)
    with open(f'{data_path}/test-queries.pkl', 'rb') as f:
        queries = pickle.load(f)
    
    return (queries, easy_answers, hard_answers)


def calculate_average_answers(queries, easy_answers, hard_answers):
    total_answers = 0
    total_queries = 0

    for query_structure in queries:

        query_set = queries[query_structure]
        for query in query_set:
            num_answers = len(easy_answers[query]) + len(hard_answers[query])
            total_answers += num_answers
            total_queries += 1

    if total_queries > 0:
        average_answers = total_answers / total_queries
    else:
        average_answers = 0

    return average_answers


if __name__=="__main__":
    args = parse_args()
    queries, easy_answers, hard_answers = load_data(args.data_path)

    avg_num_answers = calculate_average_answers(queries, easy_answers, hard_answers)

    print(avg_num_answers)
