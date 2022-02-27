from elasticsearch import Elasticsearch
from elasticsearch import helpers


class ES():
    def __init__(self):
        super().__init__()
        self.instance = Elasticsearch("localhost:9200", timeout = 100)
        self.index = "method_records"

    def insert_with_id(self, _id, record):
        return self.instance.create(index=self.index, id=_id, body=record)

    def delete_withid(self, _id):
        self.instance.delete(index=self.index, id=_id)

    def get_with_id(self, _id):
        data = self.instance.get(index=self.index, id=_id)
        if data['found'] == True:
            return data['_source']
        else:
            return {
                'index': _id,
                'beginline': -1,
                'endline': -1,
                'ast': {},
                'features': []
            }

    def update_withid(self, _id, _record):
        self.instance.update(index=self.index, id=_id, body=_record)

    def records_count(self):
        return self.instance.count(index=self.index)["count"]

    def mget_records_with_ids(self, ids):
        if len(ids) > 0:
            data = self.instance.mget(
                index=self.index, body={'ids': ids})['docs']
            records = []
            for d in data:
                if d["found"] == True:
                    records.append(d["_source"])
                else:
                    records.append({
                        'index': d['_id'],
                        'beginline': -1,
                        'endline': -1,
                        'ast': {},
                        'features': []
                    })
            return records
        else:
            return []

    def set_mapping(self):
        mapping = {
            "mappings": {
                "properties": {
                    "path": {
                        "type": "text"
                    },
                    "class": {
                        "type": "text"
                    },
                    "method": {
                        "type": "text"
                    },
                    "beginline": {
                        "type": "integer"
                    },
                    "endline": {
                        "type": "integer"
                    },
                    "index": {
                        "type": "integer"
                    },
                    "features": {
                        "type": "integer"  # array
                    },
                    "ast": {
                        "type": "flattened",
                        "index": False
                    }
                }
            }
        }
        self.instance.indices.delete(index=self.index, ignore=404)
        response = self.instance.indices.create(index=self.index, body=mapping)
        if 'acknowledged' in response:
            if response['acknowledged'] == True:
                print("INDEX MAPPING SUCCESS FOR INDEX:", response['index'])
        # catch API error response
        elif 'error' in response:
            print("ERROR:", response['error']['root_cause'])
            print("TYPE:", response['error']['type'])

    def bulk(self, actions):
        with open("elastic_err.log", "w") as f:
            for success, info in helpers.parallel_bulk(client=self.instance, actions=actions, raise_on_error=False):
                if not success:
                    f.write(str(info))
                    f.write('\n')
                    record = {
                        'index': int(info['create']['_id']),
                        'beginline': -1,
                        'endline': -1,
                        'ast': {},
                        'features': []
                    }
                    self.insert_with_id(record['index'], record)

    
    def refresh(self):
        self.instance.indices.refresh(index=self.index)


if __name__ == "__main__":
    es = ES()
    print(es.records_count())
