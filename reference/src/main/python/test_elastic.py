from elasticsearch import Elasticsearch
import ujson
import logging
import sys

class ES():
    def __init__(self):
        super().__init__()
        self.instance = Elasticsearch("localhost:9200")
        self.index = "method_records"

    def insert_with_id(self, _id, record):
        self.instance.create(index = self.index, id = _id, body = record)

    def delete_withid(self, _id ):
        self.instance.delete(index = self.index, id = _id)
    
    def get_with_id(self, _id):
        return self.instance.get(index = self.index, id = _id)["_source"]

    def update_withid(self, _id, _record):
        self.instance.update(index = self.index, id = _id, body = _record)

    def records_count(self):
        return self.instance.count(index = self.index)["count"]

    def mget_records_with_ids(self, ids):
        if len(ids) > 0:
            data = self.instance.mget(index = self.index, body={'ids' : ids})['docs']
            records = [d["_source"] for d in data]
            return records
        else:
            return []

    def set_mapping(self):
        mapping = {
            "mappings":{
                "properties":{
                    "path" : {
                        "type" : "text"
                    },
                    "class":{
                        "type" : "text"
                    },
                    "method":{
                        "type" : "text"
                    },
                    "beginline":{
                        "type": "integer"
                    },
                    "endline":{
                        "type":"integer"
                    },
                    "index":{
                        "type":"integer"
                    },
                    "feature":{
                        "type":"integer" #array
                    },
                    "ast":{
                        "type" : "flattened",
                        "index": False
                    }
                }
            }
        }
        self.instance.indices.delete(index = self.index, ignore=404)
        response = self.instance.indices.create(index = self.index, body=mapping)
        if 'acknowledged' in response:
            if response['acknowledged'] == True:
                print("INDEX MAPPING SUCCESS FOR INDEX:", response['index'])
        # catch API error response
        elif 'error' in response:
            print("ERROR:", response['error']['root_cause'])
            print("TYPE:", response['error']['type'])


def insert_features_json(es, path):
    es.instance.indices.delete(index=es.index)
    with open(path, "r") as f:
        for line in f:
            record = ujson.loads(line)
            es.insert_with_id(record["index"], record)
            logging.info(f"inserted {record['index']} records")


if __name__ == "__main__":
    es = ES()
    json = es.get_with_id(0)
    import cpp_module
    print(sys.path)
    cpp_module.test_json(json)
     
