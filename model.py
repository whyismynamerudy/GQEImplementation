"""
GQE model.
"""
from collections import defaultdict
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm


class CenterIntersection(nn.Module):
    # Geometric intersection operator implementation
    def __init__(self, embed_dim):
        super(CenterIntersection, self).__init__()
        self.embed_dim = embed_dim
        self.layer1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.layer2 = nn.Linear(self.embed_dim, self.embed_dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        out = F.relu(self.layer1(embeddings))  # (num_conj, dim)
        attention = F.softmax(self.layer2(out), dim=0)  # (num_conj, dim)
        embedding = torch.sum(attention * embeddings, dim=0)

        return embedding


class GQE(nn.Module):
    def __init__(self, num_entities, num_relations, embed_dim, gamma, query_name_dict):
        super(GQE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embed_dim = embed_dim
        self.query_names = query_name_dict

        self.gamma = nn.Parameter(
            torch.tensor([gamma]),
            requires_grad=False
        )

        self.entity_embedding = nn.Embedding(self.num_entities,
                                             self.embed_dim)

        self.relation_embedding = nn.Embedding(self.num_relations,
                                               self.embed_dim)

        self.intersection = CenterIntersection(embed_dim)

        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

    def forward(self, positive_sample, negative_sample, batch_queries_dict, batch_idx_dict, device):
        all_intersection_embeddings, all_idx = [], []
        all_union_intersection_embeddings, all_union_idx = [], []

        for qs in batch_queries_dict:
            if 'u' in self.query_names[qs]:
                transformed_query, transformed_qs = self.transform_union_query_and_structure(batch_queries_dict[qs], qs)
                q_intersect, _ = self.embed_query_vec(transformed_query, transformed_qs, 0)
                all_union_intersection_embeddings.append(q_intersect)
                all_union_idx.extend(batch_idx_dict[qs])
            else:
                q_intersect, _ = self.embed_query_vec(batch_queries_dict[qs], qs, 0)
                all_intersection_embeddings.append(q_intersect)
                all_idx.extend(batch_idx_dict[qs])

        if len(all_intersection_embeddings) > 0:
            all_intersection_embeddings = torch.cat(all_intersection_embeddings, dim=0).unsqueeze(1)
        if len(all_union_intersection_embeddings) > 0:
            all_union_intersection_embeddings = torch.cat(all_union_intersection_embeddings, dim=0).unsqueeze(1)
            all_union_intersection_embeddings = all_union_intersection_embeddings.view(
                all_union_intersection_embeddings.size(0) // 2, 2, 1, -1)

        if positive_sample is not None:
            if len(all_intersection_embeddings) > 0:
                positive_embeddings = torch.index_select(self.entity_embedding.weight, dim=0,
                                                         index=positive_sample[all_idx]).unsqueeze(1)
                # print(positive_embeddings.shape, all_intersection_embeddings.shape)
                positive_logit = self.logit(positive_embeddings, all_intersection_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(device)

            if len(all_union_intersection_embeddings) > 0:
                positive_embeddings = torch.index_select(self.entity_embedding.weight, dim=0,
                                                         index=positive_sample[all_union_idx]).unsqueeze(1).unsqueeze(1)
                positive_union_logit = \
                    torch.max(self.logit(positive_embeddings, all_union_intersection_embeddings), dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(device)

            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if negative_sample is not None:
            if len(all_intersection_embeddings) > 0:
                reg = negative_sample[all_idx]
                negative_embeddings = torch.index_select(self.entity_embedding.weight, dim=0, index=reg.view(-1)).view(
                    reg.size(0), reg.size(1), -1)
                negative_logit = self.logit(negative_embeddings, all_intersection_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(device)

            if len(all_union_intersection_embeddings) > 0:
                reg = negative_sample[all_union_idx]
                negative_embeddings = torch.index_select(self.entity_embedding.weight, dim=0, index=reg.view(-1)).view(
                    reg.size(0), 1, reg.size(1), -1)
                negative_union_logit = \
                    torch.max(self.logit(negative_embeddings, all_union_intersection_embeddings), dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(device)

            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, all_idx + all_union_idx

    # From https://github.com/snap-stanford/KGReasoning/blob/main/models.py#L243
    def embed_query_vec(self, queries, query_structure, idx):
        """
        Iterative embed a batch of queries with same structure using GQE
        queries: a flattened batch of queries
        """
        all_relation_flag = True
        for ele in query_structure[-1]:  # whether the current query tree has merged to one branch and only need to do
            # relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                # embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                embedding = F.embedding(queries[:, idx], self.entity_embedding.weight)
                idx += 1
            else:
                embedding, idx = self.embed_query_vec(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert False, "vec cannot handle queries with negation"
                else:
                    # r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    r_embedding = F.embedding(queries[:, idx], self.relation_embedding.weight)
                    embedding += r_embedding
                idx += 1
        else:
            embedding_list = []
            for i in range(len(query_structure)):
                embedding, idx = self.embed_query_vec(queries, query_structure[i], idx)
                embedding_list.append(embedding)
            embedding = self.intersection(torch.stack(embedding_list))

        return embedding, idx

    def transform_union_query_and_structure(self, queries, qs):
        query_to_return, query_structure_to_return = None, None
        if self.query_names[qs] == '2u-DNF':
            queries = queries[:, -1]
            query_structure_to_return = ('e', ('r',))
        elif self.query_names[qs] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1),
                                 torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
            query_structure_to_return = ('e', ('r', 'r'))

        query_to_return = torch.reshape(queries, [queries.shape[0] * 2, -1])
        return query_to_return, query_structure_to_return

    def logit(self, entity_emb, query_emb):
        return self.gamma - torch.norm(entity_emb - query_emb, p=1, dim=-1)

    @staticmethod
    def train_step(model, optimizer, positives, negatives, flattened_queries, query_structures, device):
        optimizer.zero_grad()

        positives.to(device)
        negatives.to(device)

        batch_queries_dict, batch_idx_dict = defaultdict(list), defaultdict(list)

        for i, query in enumerate(flattened_queries):  # group queries with the same structure
            batch_queries_dict[query_structures[i]].append(query)
            batch_idx_dict[query_structures[i]].append(i)

        for qs in batch_queries_dict:
            batch_queries_dict[qs] = torch.LongTensor(batch_queries_dict[qs]).to(device)

        positive_logit, negative_logit, _ = model(positives, negatives, batch_queries_dict, batch_idx_dict, device)

        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)

        positive_sample_loss = -positive_score.mean()
        negative_sample_loss = -negative_score.mean()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        loss.backward()
        optimizer.step()

        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }
        return log

    @staticmethod
    def evaluate(model, answers, dataloader, device):
        model.eval()

        (easy_answers, hard_answers) = answers
        logs = defaultdict(list)

        with torch.no_grad():
            for (negatives, flattened_queries, queries, query_structure) in tqdm(dataloader):
                batch_queries_dict, batch_idx_dict = defaultdict(list), defaultdict(list)

                for i, query in enumerate(queries):
                    batch_queries_dict[query_structure[i]].append(query)
                    batch_idx_dict[query_structure[i]].append(i)

                for qs in batch_queries_dict:
                    batch_queries_dict[qs] = torch.LongTensor(batch_queries_dict[qs]).to(device)

                negatives.to(device)

                _, negative_logit, idx = model(None, negatives, batch_queries_dict, batch_idx_dict, device)

                queries = [queries[i] for i in idx]
                query_structures = [query_structure[i] for i in idx]

                sorted_logits = torch.argsort(negative_logit, dim=1, descending=True)
                ranked_logits = sorted_logits.clone().to(torch.float)

                ranked_logits = ranked_logits.scatter_(1,
                                                       sorted_logits,
                                                       torch.arange(model.num_entities).to(torch.float).repeat(
                                                           sorted_logits.size(0), 1)).to(device)

                for idx, (i, query, query_structure) in enumerate(zip(sorted_logits[:, 0], queries, query_structures)):
                    hard_answer, easy_answer = hard_answers[query], easy_answers[query]
                    num_hard, num_easy = len(hard_answer), len(easy_answer)

                    curr_ranking = ranked_logits[idx, list(easy_answer) + list(hard_answer)]
                    curr_ranking, indices = torch.sort(curr_ranking)

                    masks = indices >= num_easy
                    answer_list = torch.arange(num_hard + num_easy).to(torch.float).to(device)

                    curr_ranking = curr_ranking - answer_list + 1
                    curr_ranking = curr_ranking[masks]

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

        return metrics

        # try:
        #     positives, negatives, flattened_queries, query_structures = next(iter(dataloader))
        # except StopIteration:
        #     # If the dataloader is exhausted, reinitialize it
        #     dataloader = iter(dataloader)
        #     positives, negatives, flattened_queries, query_structures = next(dataloader)
