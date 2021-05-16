import os
import logging
import pickle
import random
import ujson
import argparse
import linecache
import config
from featurize import Vocab, collect_features_as_list, counter_vectorize
from recommend import print_similar_and_completions
from elastic_api import ES
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import subprocess
from typing import List, NoReturn, Tuple, Dict, Optional, TypeVar
from multiprocessing import Pool


logging.basicConfig(level=logging.DEBUG)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--corpus",
        action="store",
        dest="corpus",
        default=None,
        help="Process raw ASTs, featurize, and store in the working directory.",
    )
    parser.add_argument(
        "-d",
        "--working-dir",
        action="store",
        dest="working_dir",
        default="/Users/huangzhibin/Downloads/aroma-paper-artifacts-master/reference/dataset/tmpout1000",
        help="Working directory.",
        required=False,
    )
    parser.add_argument(
        "-f",
        "--file-query",
        action="append",
        dest="file_query",
        default=[],
        help="File containing the query code as JAVA.",
    )
    parser.add_argument(
        "-i",
        "--index-query",
        type=int,
        action="store",
        dest="index_query",
        default=None,
        help="Index of the query AST in the corpus.",
    )
    parser.add_argument(
        "-t",
        "--testall",
        dest="testall",
        action="store_true",
        default=False,
        help="Sample config.N_SAMPLES snippets and search.",
    )
    options = parser.parse_args()
    logging.info(options)
    return options


# 从records中随机选出n个record（删减过的）
def sample_n_records(n, total_len):
    ret_indices = []
    ret_records = []
    random.seed(config.SEED)
    for j in range(10000):
        if len(ret_indices) < n:
            i = random.randint(0, total_len - 1)
            if not (i in ret_indices):
                record = get_record_part(get_record(i))
                if record != None:
                    ret_indices.append(i)
                    ret_records.append(record)
        else:
            logging.info("Sampled {len} records".format(len = len(ret_indices)))
            return (ret_indices, ret_records)
    logging.info("Sampled {len} records".format(len = len(ret_indices)))
    return (ret_indices, ret_records)


def get_sub_ast_aux(ast, beginline, stop=False):
    if isinstance(ast, list):
        if stop:
            return (stop, None)
        else:
            ret = []
            for elem in ast:
                (stop, tmp) = get_sub_ast_aux(elem, beginline, stop)
                if tmp != None:
                    ret.append(tmp)
            if len(ret) >= 2:
                return (stop, ret)
            else:
                return (True, None)
    elif isinstance(ast, dict) and "label" not in ast:
        if (
                "leaf" not in ast
                or not ast["leaf"]
                or (not stop and ast["line"] - beginline < config.SAMPLE_METHOD_MAX_LINES)
        ):
            return (stop, ast)
        else:
            return (True, None)
    else:
        return (stop, ast)


def copy_record_with_ast(record, ast):
    ret = dict(record)
    ret["ast"] = ast
    return ret


# 获取record的删减版，用于测试
def get_record_part(record):
    n_lines = record["endline"] - record["beginline"]
    if n_lines < config.SAMPLE_METHOD_MIN_LINES:
        return None
    else:
        (_, ast) = get_sub_ast_aux(record["ast"], record["beginline"])
        if ast == None:
            return None
        else:
            ret = copy_record_with_ast(record, ast)
            ret["features"] = collect_features_as_list(ast, False, False, vocab)
            return ret


# 从records库拿出第idx个record，将其删减后输入，进行测试，看能否得到原record
def test_record_at_index(idx):
    record = get_record(idx)
    record = get_record_part(record)
    if record != None:
        print_similar_and_completions(
            record, get_records, vectorizer, counter_matrix)


def featurize_and_test_record(record: dict) -> List[str]:
    record = featurize_query_record(record)
    if len(record["features"]) > 0:
        return print_similar_and_completions(record, get_records, vectorizer, counter_matrix)


def featurize_query_record(record: dict) -> dict:
    record["features"] = collect_features_as_list(
        record["ast"], False, False, vocab)
    record["index"] = -1
    return record


def test_all(total_len):
    N = config.N_SAMPLES
    (sampled_indices, sampled_records) = sample_n_records(N, total_len)
    for k, record in enumerate(sampled_records):
        print(f"{k}: ", end="")
        print_similar_and_completions(
            record, get_records, vectorizer, counter_matrix)


# 读 features.json
def read_all_records(rpath):
    ret = []
    with open(rpath, "r") as f:
        for record in f:
            r = ujson.loads(record)
            ret.append(r)
    return ret


def load_matrix(counter_path):
    with open(counter_path, "rb") as outf:
        (vectorizer, counter_matrix) = pickle.load(outf)
        logging.info("Read vectorizer and counter matrix.")
    return (vectorizer, counter_matrix)


def setup(ast_path):
    global vectorizer
    global counter_matrix
    global vocab
    os.makedirs(options.working_dir, exist_ok=True)
    vocab = Vocab.load(options.working_dir)
    config.g_vocab = vocab
    if ast_path is not None:
        run_buildIndex(ast_path, options.working_dir)
    (vectorizer, counter_matrix) = load_matrix(
        os.path.join(options.working_dir, config.TFIDF_FILE)
    )


def get_json_by_line_from_file(line):
    path = os.path.join(options.working_dir, config.FEATURES_FILE)
    content = linecache.getline(path, line + 1)
    return ujson.loads(content)


def get_record(id: int):
    return es_instance.get_with_id(id)


def get_records(ids: List[int]):
    return es_instance.mget_records_with_ids(ids)


def insert_record(id: int, record: dict):
    return es_instance.insert_with_id(id, record)


def get_recordQuantity():
    return es_instance.records_count()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins='http://localhost:8060',
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/test")
def run_pipeline(path: str, line: int) -> dict:
    command = 'java -jar /Users/huangzhibin/code-recommendation/aroma-paper-artifacts/reference/target/ANTLR4SimpleAST-1.0-SNAPSHOT-jar-with-dependencies.jar compilationUnit stdout {inputpath}'.format(
        inputpath=path)
    complete = subprocess.run(
        command, check=True, text=True, capture_output=True, shell=True)
    records = complete.stdout.strip().split('\n')
    for record in records:
        record = ujson.loads(record)
        if record['beginline'] <= line and line <= record['endline']:
            results = featurize_and_test_record(record)
            break
    else:
        results = ['// 请将光标移至方法所在行。']
    return {"recommendation": results}


def run_buildIndex(path: str, working_dir: str) -> NoReturn:
    id = config.RECORD_QUANTITY
    def get_java(path: str, path_set: set) -> NoReturn:
        if os.path.exists(path):
            for f in os.listdir(path):
                wholepath = os.path.join(path, f)
                if os.path.isdir(wholepath):
                    get_java(wholepath, path_set)
                elif os.path.isfile(wholepath):
                    if wholepath.endswith(".java") and 'test' not in wholepath.lower():
                        relpath = os.path.relpath(wholepath)
                        path_set.add(relpath)
                        logging.info("Path set size: " +
                                     str(len(path_set)) + ',' + relpath)

    # if os.path.exists(os.path.join(working_dir, "records_path.pkl")):
    #     with open(os.path.join(working_dir, "records_path.pkl"), "rb") as f:
    #         path_set = pickle.load(f)
    # else:
    #     path_set = set()
    # get_java(path, path_set)
    # with open(os.path.join(working_dir, "records_path.pkl"), "wb") as out:
    #     pickle.dump(path_set, out)
    #     logging.info("Saved the records source path")
    # with open(os.path.join(working_dir, "records_path.txt"), "w") as f:
    #     for i, path in enumerate(path_set):
    #         f.write(path)
    #         f.write('\n')
    # inputpath = os.path.join(working_dir, "records_path.txt")
    # outputpath = os.path.join(working_dir, "ast.json")
    # command = 'java -jar /Users/huangzhibin/code-recommendation/aroma-paper-artifacts/reference/target/ANTLR4SimpleAST-1.0-SNAPSHOT-jar-with-dependencies.jar compilationUnit {outputpath} {inputpath}'.format(outputpath = outputpath, inputpath = inputpath)
    # subprocess.run(command, cwd = os.getcwd(), check= True, shell = True)

    def run_featurize(id: int):
        with open(os.path.join(working_dir, "ast.json"), "r") as f:
            for line in f:
                record = ujson.loads(line)
                record["features"] = collect_features_as_list(
                    record["ast"], True, False, vocab)
                record["index"] = id
                yield {
                    '_op_type': 'create',
                    '_index': 'method_records',
                    '_id': id,
                    '_source': record
                }
                id += 1
                #logging.info("Has featurized: " + str(id))
    # 并行批量插入
    # es_instance.bulk(run_featurize(id))
    # logging.info("Featurized {i} methods".format(i=id))
    # vocab.dump(working_dir)
    # logging.info("Dumped feature vocab.")
    counter_vectorize(
            get_records,
            os.path.join(options.working_dir, config.TFIDF_FILE),
    )

def collect_features_as_list_wapperForCpp(ast, is_init: bool, is_counter: bool):
    return collect_features_as_list(ast, is_init, is_counter, config.g_vocab)


if __name__ == "__main__":
    options = parse_args()
    vectorizer = None
    counter_matrix = None
    vocab = None
    es_instance = ES()
    setup(options.corpus)
    #config.RECORD_QUANTITY = get_recordQuantity()
    if options.index_query is not None:
        test_record_at_index(options.index_query)
    elif len(options.file_query) > 0:
        with open(options.file_query, "r") as f:
            for line in f:
                obj = ujson.loads(line)
                featurize_and_test_record(obj)
    elif options.testall:
        config.TEST_ALL = True
        test_all(total_len=config.RECORD_QUANTITY)
    else:
        uvicorn.run(app, host="127.0.0.1", port=8000)
