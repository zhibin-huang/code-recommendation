import os
import logging
import pickle
import random
import ujson
import argparse
import linecache
import config
from featurize import Vocab, collect_features_as_list, counter_vectorize, featurize_ast_file, featurize_query_record
from recommand import print_similar_and_completions
from test_elastic import ES
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import subprocess

options = None
vectorizer = None
counter_matrix = None
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
        default = "/Users/huangzhibin/Downloads/aroma-paper-artifacts-master/reference/dataset/tmpout1000",
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
                record = get_record_part(get_record_by_id(i))
                if record != None:
                    ret_indices.append(i)
                    ret_records.append(record)
        else:
            logging.info("Sampled records")
            return (ret_indices, ret_records)
    logging.info("Sampled records")
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
            ret["features"] = collect_features_as_list(ast, False, False)
            return ret


# 从records库拿出第idx个record，将其删减后输入，进行测试，看能否得到原record
def test_record_at_index(idx):
    record = get_record_by_id(idx)
    record = get_record_part(record)
    if record != None:
        print_similar_and_completions(record, get_record_by_id, vectorizer, counter_matrix)


def featurize_and_test_record(record):
   record = featurize_query_record(record)
   if len(record["features"]) > 0:
       return print_similar_and_completions(record, get_records_by_ids_from_es, vectorizer, counter_matrix)


def test_all(total_len):
    N = config.N_SAMPLES
    (sampled_indices, sampled_records) = sample_n_records(N, total_len)
    for k, record in enumerate(sampled_records):
        print(f"{k}: ", end="")
        print_similar_and_completions(record, get_records_by_ids_from_es, vectorizer, counter_matrix)


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


def setup(ast_file, get_records_by_ids):
    global vectorizer
    global counter_matrix
    os.makedirs(options.working_dir, exist_ok=True)
    if ast_file is None:
        Vocab.load(options.working_dir)
        (vectorizer, counter_matrix) = load_matrix(
            os.path.join(options.working_dir, config.TFIDF_FILE)
        )
    else:
        Vocab.load(options.working_dir, init=True)
        featurize_ast_file(
            ast_file, options.working_dir
        )
        counter_vectorize(
            get_records_by_ids,
            os.path.join(options.working_dir, config.TFIDF_FILE),
        )


def get_json_by_line_from_file(line):
    path = os.path.join(options.working_dir, config.FEATURES_FILE)
    content = linecache.getline(path, line + 1)
    return ujson.loads(content)


def get_record_by_id_from_es(id):
    return es_instance.get_with_id(id)


def get_records_by_ids_from_es(ids):
    return es_instance.mget_records_with_ids(ids)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = 'http://localhost:8060',
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

@app.get("/")
async def startTest():
    return {"message : hello!"}

@app.get("/test")
def run_pipeline(path: str, line : int):
    command = 'java -jar /Users/huangzhibin/Downloads/aroma-paper-artifacts-master/reference/target/ANTLR4SimpleAST-1.0-SNAPSHOT-jar-with-dependencies.jar compilationUnit stdout {inputpath}'.format(inputpath = path)
    complete = subprocess.run(command, check= True, text = True, capture_output = True, shell = True)
    records = complete.stdout.strip().split('\n')
    for record in records:
        record = ujson.loads(record)
        if record['beginline'] <= line and line <= record['endline']:
            results = featurize_and_test_record(record)
            break
    else:
        results = ['// 请将光标移至方法所在行。']
    return {"recommendation" : "".join(results)}
    
    


if __name__ == "__main__":
    options = parse_args()
    es_instance = ES()
    get_record_by_id = get_record_by_id_from_es
    setup(options.corpus, get_records_by_ids_from_es)

    if options.index_query is not None:
        test_record_at_index(options.index_query)
    elif len(options.file_query) > 0:
         featurize_and_test_record(options.file_query, options.keywords)
    elif options.testall:
        config.TEST_ALL = True
        test_all(total_len=config.RECORD_QUANTITY)
    #uvicorn.run(app, host = "127.0.0.1", port = 8000)