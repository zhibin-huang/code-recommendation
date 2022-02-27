import logging
from collections import Counter
from sklearn.metrics.pairwise import linear_kernel

import config
from featurize import feature_list_to_doc
from cpp_module import jaccard, prune_last_jd, prune_parallel

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)

# ast还原成代码


def ast_to_code_aux(ast, token_list):
    if isinstance(ast, list):
        for elem in ast:
            ast_to_code_aux(elem, token_list)
    elif isinstance(ast, dict) and "label" not in ast:
        token_list.append(ast["leading"])
        token_list.append(ast["token"])


def ast_to_code_collect_lines(ast, line_list):
    if isinstance(ast, list):
        for elem in ast:
            ast_to_code_collect_lines(elem, line_list)
    elif isinstance(ast, dict) and "label" not in ast:
        if "line" in ast:
            line_list.append(ast["line"])


def ast_to_code_print_lines(ast, line_list, token_list):
    if isinstance(ast, list):
        for elem in ast:
            ast_to_code_print_lines(elem, line_list, token_list)
    elif isinstance(ast, dict) and "label" not in ast:
        if "line" in ast and ast["line"] in line_list:
            if len(token_list) > 0 and token_list[-1] == "//":
                token_list.append(" your code ...\n")
            token_list.append(ast["leading"])
            token_list.append(ast["token"])
        else:
            if len(token_list) > 0 and token_list[-1] != "//":
                token_list.append("\n")
                token_list.append("//")


# def print_features(fstr):
    # print(" ".join([vocab.get_word(int(k)) for k in fstr]))


# 计算点积
def my_similarity_score(M1, M2):
    return linear_kernel(M1, M2)


# def copy_leaf_dummy(ast):
    # return {"token": " ... ", "leading": ast["leading"], "trailing": ast["trailing"]}


# 从ast中获得含有leaf_features的子树（砍掉已成为none的token）
# def prune_ast(ast, leaf_features, leaf_idx):
    # if isinstance(ast, list):
    #     no_leaf = True
    #     ret = []
    #     for elem in ast:
    #         (flag, tmp, leaf_idx) = prune_ast(elem, leaf_features, leaf_idx)
    #         ret.append(tmp)
    #         no_leaf = no_leaf and flag
    #     # 子树的叶结点token都被砍完了（flag每次返回true），则丢弃整棵子树
    #     # 否则返回修剪好的ast
    #     if no_leaf:
    #         return (True, None, leaf_idx)
    #     else:
    #         return (False, ret, leaf_idx)
    # elif isinstance(ast, dict) and "label" not in ast:
    #     if "leaf" in ast and ast["leaf"]:
    #         leaf_idx += 1
    #         # 去除ast中要被砍掉的token，被砍的token变成 "..."
    #         if len(leaf_features[leaf_idx - 1])== 0:
    #             return (True, copy_leaf_dummy(ast), leaf_idx)
    #         else:
    #             return (False, ast, leaf_idx)
    #     else:
    #         return (True, ast, leaf_idx)
    # else:
    #     return (True, ast, leaf_idx)


# #Jaccard 相似系数
# def jaccard(counter1, counter2):
    # return sum((counter1 & counter2).values()) / sum((counter1 | counter2).values())


# def copy_record(record2, ast, features):
    # ret = dict(record2)
    # ret["ast"] = ast
    # ret["features"] = features
    # ret["index"] = -1
    # return ret


# 测试用
fail = 0
def print_match_index(query_record, candidate_records):
    ret = -1
    i = 0
    for (candidate_record, score, pruned_record, pruned_score) in candidate_records:
        if query_record["index"] == candidate_record["index"]:
            ret = i
        i += 1
    if ret < 0:
        global fail
        fail += 1
        print("Failed to match original method. " + str(fail))
    elif ret > 0:
        print(f"Matched original method. Rank = {ret}")
    else:
        print(f"Matched original method perfectly.")


# 取相似度大于min_similarity_score 的前 num_similars 个候选结果索引
def find_indices_similar_to_features(
        vectorizer, counter_matrix, feature_lists_doc, num_similars, min_similarity_score
):
    doc_counter_vector = vectorizer.transform(feature_lists_doc)
    len = my_similarity_score(
        doc_counter_vector, doc_counter_vector).flatten()[0]
    cosine_similarities = my_similarity_score(
        doc_counter_vector, counter_matrix
    ).flatten()
    related_docs_indices = [
        i
        for i in cosine_similarities.argsort()[::-1]
        if cosine_similarities[i] > min_similarity_score * len
    ][0:num_similars]
    return [(j, cosine_similarities[j]) for j in related_docs_indices]


# 将record2修剪成与records集合最相似（以jaccard距离度量）
# def prune_last_jd(records, record2):
    # other_features = [Counter(record["features"]) for record in records]
    # leaves_features_count = collect_features_as_list(record2["ast"], False, True)
    # # 贪心算法选择的近似最优叶结点
    # out_features = [None] * len(leaves_features_count)
    # current_features_count = Counter()
    # for features1 in other_features:
    #     score = jaccard(features1, current_features_count)
    #     done = False
    #     # 逐渐向current_features添加能使两树feature集合交集最大的feature
    #     while not done:
    #         max = score
    #         max_idx = None
    #         i = 0
    #         for leaf_features_count in leaves_features_count:
    #             if leaf_features_count is not None:
    #                 new_features_count = current_features_count + leaf_features_count
    #                 tmp = jaccard(features1, new_features_count)
    #                 if tmp > max:
    #                     max = tmp
    #                     max_idx = i
    #             i += 1
    #         if max_idx is not None:
    #             score = max
    #             out_features[max_idx] = leaves_features_count[max_idx]
    #             current_features_count = current_features_count + leaves_features_count[max_idx]
    #             leaves_features_count[max_idx] = None
    #         else:
    #             done = True
    # if isinstance(records, dict):
    #     records = [records]
    # #print("enter prune_last_jd")
    # out_features = get_out_features(records, record2)
    # pruned_ast = prune_ast(record2["ast"], out_features, leaf_idx=0)[1]
    # pruned_features = collect_features_as_list(pruned_ast, False, False)
    # return copy_record(record2, pruned_ast, pruned_features)

# ast还原成代码
def ast_to_code(tree):
    token_list = []
    ast_to_code_aux(tree, token_list)
    token_list.append("\n")
    return "".join(token_list)


def ast_to_code_with_full_lines(tree, fulltree):
    line_list = []
    ast_to_code_collect_lines(tree, line_list)
    token_list = []
    ast_to_code_print_lines(fulltree, line_list, token_list)
    token_list.append("\n")
    return "".join(token_list)


# 经过light-weight serch，prune and rerank 得到的候选集
def find_similar(
        query_record,
        get_records_by_ids,
        vectorizer,
        counter_matrix,
        num_similars,
        min_similarity_score,
        min_pruned_score,
):
    #print("Query features: ")
    # print_features(query_record["features"])
    # 获取候选结果（light-weight search）
    similars = find_indices_similar_to_features(
        vectorizer,
        counter_matrix,
        [feature_list_to_doc(query_record["features"])],
        num_similars,
        min_similarity_score,
    )
    candidate_records = []
    # 修剪候选片段，重新排序 （prune and rerank）
    if len(similars) > 0:
        similar_records = get_records_by_ids(
            [idx for (idx, score) in similars])
        scores = [score for (idx, score) in similars]
        # for i, similar_record in enumerate(similar_records):
        #     pruned_record = prune_last_jd([query_record], similar_record)
        #     pruned_score = jaccard(Counter(query_record["features"]), Counter(pruned_record["features"]))
        #     if pruned_score > min_pruned_score:
        #         candidate_records.append((similar_record, scores[i], pruned_record, pruned_score))
        candidate_records = prune_parallel(
            query_record, min_pruned_score, similar_records, scores)
        candidate_records = sorted(
            candidate_records, key=lambda v: v[3], reverse=True)
        logging.info("# of similar snippets = {len}".format(
            len=len(candidate_records)))
    return candidate_records

# records列表内的features交集大小


def find_similarity_score_features_set_un(records):
    features_as_counters = []
    for record in records:
        features_as_counters.append(Counter(record["features"]))
    intersection = None
    for counter in features_as_counters:
        if intersection is None:
            intersection = counter
        else:
            intersection = intersection & counter
    return sum(intersection.values())


# 得到最终推荐的records
def cluster_and_intersect(
        query_record, candidate_records, top_n, threshold1, threshold2
):
    len_candidate = len(candidate_records)
    clustered_records = []
    if len_candidate > 0:
        ret = []
        acc = []
        # 初始一个record作为一个cluster
        for i in range(len_candidate):
            cs = len(candidate_records[i][0]["features"])  # 原始record
            csq = len(candidate_records[i][2]["features"])  # pruned record
            # s(τ) = csq(τ)/|F(Prune(F(q), N2(i1)) )| = 1
            # l(τ) = cs(τ) / csq(τ) > 1.5
            if cs > csq * threshold2:
                ret.append([i])
        changed = True
        while changed:
            tmp = []
            changed = False
            for tuple in ret:
                kmax = None
                maxscore = 0
                pruned_record_list = [candidate_records[i][2] for i in tuple]
                original_record_list = [candidate_records[i][0] for i in tuple]
                for k in range(tuple[-1] + 1, len_candidate):
                    pruned_record_list.append(candidate_records[k][2])
                    original_record_list.append(candidate_records[k][0])
                    # 该tuple第一个元素（pruned record）的feature数量
                    qlen = len(pruned_record_list[0]["features"])
                    # 该tuple中所有pruned_record的feature交集大小 csq(τ)
                    csq = find_similarity_score_features_set_un(
                        pruned_record_list)
                    # s(τ) = csq(τ)/|F(Prune(F(q), N2(i1)) )|
                    # 该tuple中所有pruned_records与第一个record的相似度，度量tuple内聚度
                    pscore = csq / qlen
                    # pscore = find_similarity_score_features_set(record_list1)
                    if pscore > threshold1:
                        # 该tuple中所有原始record的feature交集大小 cs(τ)
                        cs = find_similarity_score_features_set_un(
                            original_record_list)
                        # l(τ) = cs(τ)/csq(τ)
                        # 在满足阈值的情况下，l(τ)尽量大，以满足最后结果比query长
                        if cs > threshold2 * csq and cs > maxscore:
                            kmax = k
                            maxscore = cs
                    original_record_list.pop()
                    pruned_record_list.pop()
                if kmax is not None:
                    changed = True
                    tuple.append(kmax)
                    tmp.append(tuple)
            acc.extend(tmp)
            ret = tmp
        # 第一关键字为tuple中第一个元素的index（升序），第二关键字是tuple的长度（降序）
        acc = sorted(acc, key=lambda t: t[0] * 1000 - len(t))
        if len(acc) > 0:
            for i in range(len(acc)):
                tuple = acc[i]
                is_subset = False
                for j in reversed(range(i)):
                    if jaccard(Counter(tuple), Counter(acc[j])) > 0.5:
                        is_subset = True
                        break
                if not is_subset:
                    logging.info("recommending")
                    logging.info("Pruning {len} {t}".format(
                        len=len(tuple), t=tuple))
                    pruned_record = candidate_records[tuple[0]][0]
                    # Intersect((i1,...,ij,ij+1),q)=Prune(F(N2(ij+1))∪F(q),Intersect((i1,...,ij),q))
                    for j in range(1, len(tuple)):
                        pruned_record = prune_last_jd(
                            [query_record, candidate_records[tuple[j]]
                                [0]], pruned_record
                        )
                    clustered_records.append(
                        [pruned_record, candidate_records[tuple[0]][0]])
                    if len(clustered_records) >= top_n:
                        return clustered_records
    return clustered_records


def print_similar_and_completions(query_record, get_records_by_ids, vectorizer, counter_matrix):
    results = []
    candidate_records = find_similar(
        query_record,
        get_records_by_ids,
        vectorizer,
        counter_matrix,
        config.NUM_SIMILARS,
        config.MIN_SIMILARITY_SCORE,
        config.MIN_PRUNED_SCORE
    )
    if config.CLUSTER and len(candidate_records) > 0:
        clustered_records = cluster_and_intersect(
            query_record,
            candidate_records,
            config.TOP_N,
            config.THRESHOLD1,
            config.THRESHOLD2,
        )
        id = 0
        for clustered_record in clustered_records:
            # idxs = ({clustered_record[1:]}), score = {candidate_records[clustered_record[1]][3]}")
            # clutered_record[0]: 该cluter求交集的结果，clustered_record[1] : 该cluter里的第一个元素
            results.append(ast_to_code_with_full_lines(
                clustered_record[0]["ast"], clustered_record[1]["ast"]
            ))

    if config.TEST_ALL:
        print_match_index(query_record, candidate_records)
        if config.PRINT_TEST:
            print(
                f"################ query code ################ index = {query_record['index']}"
            )
            print(ast_to_code(query_record["ast"]))
            if query_record["index"] >= 0:
                print("---------------- extracted from ---------------")
                record = get_records_by_ids([query_record["index"]])[0]
                print(ast_to_code(record["ast"]))
            print("".join(results))
    if config.PRINT_SIMILAR:
        j = 0
        for (candidate_record, score, pruned_record, pruned_score) in candidate_records:
            print(
                f"idx = {j}:------------------- similar code ------------------ index = {candidate_record['index']}, score = {score}"
            )
            print(ast_to_code(candidate_record["ast"]))
            print(
                f"------------------- similar code (pruned) ------------------ score = {pruned_score}"
            )
            print(ast_to_code(pruned_record["ast"]))
            j += 1
    print("", flush=True)
    return results
