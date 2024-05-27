"""
Helpful functions.
"""


def flatten(l):
    if isinstance(l, tuple):
        return sum(map(flatten, l), [])
    else:
        return [l]


def flatten_query(queries):
    all_queries = []
    # print("in flatten_query, keys of queries: ", queries.keys())
    for query_structure in queries:
        tmp_queries = list(queries[query_structure])
        # print("qs: {}, len of temp_queries: {}".format(query_structure, len(tmp_queries)))
        all_queries.extend([(query, query_structure) for query in tmp_queries])

    # print("len_all_queries: {}".format(len(all_queries)))
    return all_queries
