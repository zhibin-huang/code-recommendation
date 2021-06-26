import os
import ujson
import logging
import pickle
import re
from collections import OrderedDict, Counter
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, NoReturn, Tuple, Dict, Optional, TypeVar, Callable
import config


class Vocab:
    def __init__(self):
        self.vocab = OrderedDict()
        self.words = []

    def get_word(self, i):
        if i <= config.NUM_FEATURE_MIN:
            return "#UNK"
        return self.words[i - 1 - config.NUM_FEATURE_MIN]

    def add_and_get_index(self, word):
        if not (word in self.vocab):
            self.words.append(word)
            self.vocab[word] = [
                0, len(self.vocab) + 1 + config.NUM_FEATURE_MIN]
        value = self.vocab[word]
        value[0] += 1
        return value[1]

    def get_index(self, word):
        if word in self.vocab:
            return self.vocab[word][1]
        else:
            return config.NUM_FEATURE_MIN

    def dump(self, working_dir):
        with open(os.path.join(working_dir, config.VOCAB_FILE), "wb") as out:
            pickle.dump([self.vocab, self.words], out)
            logging.info(f"Dumped vocab with size {len(self.vocab)}")

    @staticmethod
    def load(working_dir):
        tmp = Vocab()
        if os.path.exists(os.path.join(working_dir, config.VOCAB_FILE)):
            with open(
                    os.path.join(working_dir, config.VOCAB_FILE), "rb"
            ) as out:
                [tmp.vocab, tmp.words] = pickle.load(out)
                logging.info(f"Loaded vocab with size {len(tmp.vocab)}")
        return tmp


Ast = TypeVar("AST", list, dict)

# 寻找ast子树最左的叶子结点
def get_leftmost_leaf(ast: Ast) -> Tuple[bool, Optional[str]]:
    if isinstance(ast, list):
        for elem in ast:
            (success, token) = get_leftmost_leaf(elem)
            if success:
                return (success, token)
    elif isinstance(ast, dict) and "label" not in ast:
        if "leaf" in ast and ast["leaf"]:
            return (True, ast["token"])
    return (False, None)


# 局部变量的位置上下文，区分不同 "#VAR"
def get_var_context(p_idx: int, p_label: str, p_ast: Ast) -> str:
    # “#.#” 特殊情况，见论文
    if p_label == "#.#":
        return get_leftmost_leaf(p_ast[p_idx + 2])[1]
    else:
        return p_label + str(p_idx)


# 向feature_list添加特征key的索引值，初始化阶段如果vocab中不存在特征key，先将其加入vocab
def append_feature_index(
    is_init: bool,
    is_counter: bool,
    key: str,
    vocab: Vocab,
    feature_list: List[int],
    c: Counter
) -> NoReturn:
    if is_init:
        n = vocab.add_and_get_index(key)
    else:
        n = vocab.get_index(key)
    if is_counter:
        if n != str(config.NUM_FEATURE_MIN):
            c[n] += 1
    else:
        feature_list.append(n)


def append_feature_pair(
        is_init: bool,
        is_counter: bool,
        key: str,
        vocab: Vocab,
        feature_list: List[int],
        leaf_features_count: List[Counter],
        sibling_idx: int,
        leaf_idx: int
) -> NoReturn:
    if is_init:
        n = vocab.add_and_get_index(key)
    else:
        n = vocab.get_index(key)
    if is_counter:
        if n != str(config.NUM_FEATURE_MIN):
            leaf_features_count[leaf_idx][n] += 1
            leaf_features_count[sibling_idx][n] += 1
    else:
        feature_list.append(n)
        feature_list.append(n)


def feature_list_to_doc(feature_list: List[int]) -> str:
    return " ".join([str(y) for y in feature_list])


def counter_vectorize(get_records_by_ids: Callable[[List[int]], dict], wpath: str) -> NoReturn:
    quantity = config.RECORD_QUANTITY
    vectorizer = CountVectorizer(min_df=1, binary=True)
    documents = []
    cookie = int(quantity / 50)
    i = 0
    while i + cookie <= quantity:
        features = [record["features"] for record in get_records_by_ids(
            [i for i in range(i, i + cookie)])]
        docs = [feature_list_to_doc(feature) for feature in features]
        documents.extend(docs)
        i += cookie
        logging.info("load {i} documents.".format(i=i))
    if i < quantity:
        features = [record["features"] for record in get_records_by_ids(
            [i for i in range(i, i + cookie)])]
        docs = [feature_list_to_doc(feature) for feature in features]
        documents.extend(docs)
        logging.info("load {i} documents.".format(i=quantity))
    counter_matrix = vectorizer.fit_transform(documents)
    logging.info("Finished Vectorizing feature documents.")
    with open(wpath, "wb") as outf:
        pickle.dump((vectorizer, counter_matrix), outf)
    logging.info("Dump vectorizer, counter matrix")


leaf_idx = 0


def collect_features_aux(
        ast: Ast,
        feature_list: List[int],
        parents: List[Tuple[int, str, Ast]],
        siblings: List[Tuple[int, str]],
        var_siblings: Dict[str, List[Tuple[int, str]]],
        leaf_features_count: List[Counter],
        is_init: bool,
        is_counter: bool,
        vocab: Vocab
) -> NoReturn:
    global leaf_idx
    # ast为list时，表示这是棵子树
    # ast[0]["label"] ：子树根结点的字符串表示，ast[1]及之后的元素为该子树的子结点
    if isinstance(ast, list):
        i = 0
        for elem in ast:
            # 形成parent信息的元组，在提取下一层结点（elem）特征时使用
            parents.append((i, ast[0]["label"], ast))
            collect_features_aux(
                elem,
                feature_list,
                parents,
                siblings,
                var_siblings,
                leaf_features_count,
                is_init,
                is_counter,
                vocab
            )
            parents.pop()
            i += 1
    # ast为dict时，表示这是一个词法记号（端结点）
    # 对叶子结点（token）进行特征提取
    elif isinstance(ast, dict) and "label" not in ast:
        if "leaf" in ast and ast["leaf"]:
            leaf_idx += 1
            is_var = False
            var_name = key = ast["token"]
            # 局部变量变为“#VAR”
            if config.IGNORE_VAR_NAMES and "var" in ast and not key[0].isupper():
                key = "#VAR"
                is_var = True
            c = None
            # 为该结点创建一个特征计数的dict
            if is_counter:
                c = Counter()
                leaf_features_count.append(c)
            # 向feature_list加入token feature的索引值
            append_feature_index(is_init, is_counter, key,
                                 vocab, feature_list, c)
            # parent feature，自下而上就近提取
            count = 0
            for (i, p, p_ast) in reversed(parents):
                if p != "(#)" and re.match("^\{#*\}$", p) is None:
                    count += 1
                    # key为p的第i个孩子
                    key2 = p + str(i) + ">" + key
                    append_feature_index(
                        is_init, is_counter, key2, vocab, feature_list, c)
                    if count >= config.N_PARENTS:
                        break

            # 局部变量的variable usage feature
            count = 0
            if not config.IGNORE_VAR_SIBLING_FEATURES and is_var:
                (p_idx, p_label, p_ast) = parents[-1]
                var_context = get_var_context(p_idx, p_label, p_ast)
                if var_context is not None:
                    if var_name not in var_siblings:
                        var_siblings[var_name] = []
                    for (var_sibling_idx, var_context_sibling) in reversed(
                            var_siblings[var_name]
                    ):
                        count += 1
                        # var_context_sibling 出现在 var_context 前面
                        key2 = var_context_sibling + ">>>" + var_context
                        #                        logging.info(f"var sibling feature {key2}")
                        append_feature_pair(
                            is_init,
                            is_counter,
                            key2,
                            vocab,
                            feature_list,
                            leaf_features_count,
                            var_sibling_idx,
                            leaf_idx - 1
                        )
                        if count >= config.N_VAR_SIBLINGS:
                            break
                    var_siblings[var_name].append((leaf_idx - 1, var_context))

            # sibling feature
            count = 0
            if not config.IGNORE_SIBLING_FEATURES:  # and not is_var:
                for (sibling_idx, sibling) in reversed(siblings):
                    count += 1
                    # sibling 在 key 之前出现
                    key2 = sibling + ">>" + key
                    append_feature_pair(
                        is_init,
                        is_counter,
                        key2,
                        vocab,
                        feature_list,
                        leaf_features_count,
                        sibling_idx,
                        leaf_idx - 1
                    )
                    if count >= config.N_SIBLINGS:
                        break
                siblings.append((leaf_idx - 1, key))


def collect_features_as_list(ast: Ast, is_init: bool, is_counter: bool, vocab: Vocab) -> list:
    feature_list: List[int] = []
    leaf_features_count: List[Counter] = []
    global leaf_idx
    leaf_idx = 0
    collect_features_aux(
        ast,
        feature_list,
        [],
        [],
        dict(),
        leaf_features_count,
        is_init,
        is_counter,
        vocab
    )
    if is_counter:
        return leaf_features_count
    else:
        return feature_list
