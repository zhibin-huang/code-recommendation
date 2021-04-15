#include <algorithm>
#include <mutex>
#include <nlohmann/json.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11_json/pybind11_json.hpp>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
namespace nl = nlohmann;
using namespace std;
using counter = unordered_map<int, int>;
using json = nl::json;
using candidate_tuple = tuple<json, double, json, double>;
using clustered_record = vector<json>;

inline double jaccard(const counter &counter1, const counter &counter2) {
  int a = 0, b = 0;
  auto j = counter1.begin();
  for (auto i = counter2.begin(); i != counter2.end(); ++i) {
    int key = i->first, value = i->second;
    // 1 和 2 都有
    if (counter1.count(key)) {
      a += std::min(counter1.at(key), value);
      b += std::max(counter1.at(key), value);
    }
    // 2 有， 1 没有
    else
      b += value;
    // 1 有， 2 没有
    if (j != counter1.end() && !counter2.count(j->first)) {
      b += j->second;
    }
    if (j != counter1.end())
      j++;
  }
  while (j != counter1.end()) {
    if (!counter2.count(j->first)) {
      b += j->second;
    }
    j++;
  }
  return 1.0 * a / b;
}

inline counter union_counter(counter counter1, counter counter2) {
  for (auto &i : counter2) {
    if (counter1.count(i.first))
      counter1[i.first] += i.second;
    else
      counter1[i.first] = i.second;
  }
  return counter1;
}

inline counter list_toCounter(const vector<int> &list) {
  counter tmp;
  for (int i : list) {
    if (tmp.count(i))
      tmp[i] = tmp[i] + 1;
    else
      tmp[i] = 1;
  }
  return tmp;
}

inline json copy_leaf_dummy(const json &ast) {
  json j;
  j["token"] = "...";
  j["leading"] = ast["leading"];
  j["trailing"] = ast["trailing"];
  return j;
}

int find_similarity_score_features_set_un(const vector<json> &records) {
  vector<counter> features_as_counters;
  size_t min_size;
  int idx = 0;
  for (int i = 0; i < records.size(); ++i) {
    counter c = list_toCounter(records[i]["features"]);
    if (i == 0)
      min_size = c.size();
    else {
      if (min_size > c.size()) {
        min_size = c.size();
        idx = i;
      }
    }
    features_as_counters.push_back(c);
  }
  int res = 0;
  for (auto i = features_as_counters[idx].begin();
       i != features_as_counters[idx].end(); i++) {
    int t = i->second;
    for (int j = 0; j < features_as_counters.size(); j++) {
      if (features_as_counters[j].count(i->first)) {
        t = min(t, features_as_counters[j][i->first]);
      }
    }
    res += t;
  }
  return res;
}

tuple<json, bool> prune_ast(const json &ast,
                            const vector<counter> &leaf_features,
                            int &leaf_idx) {
  if (ast.type() == json::value_t::array) {
    bool no_leaf = true;
    json ret;
    for (auto &elem : ast) {
      tuple<json, bool> t = prune_ast(elem, leaf_features, leaf_idx);
      ret.push_back(std::get<0>(t));
      no_leaf = no_leaf && std::get<1>(t);
    }
    if (no_leaf)
      return tuple<json, bool>(nullptr, true);
    else
      return tuple<json, bool>(ret, false);
  } else if (ast.type() == json::value_t::object &&
             ast.find("leaf") != ast.end()) {
    leaf_idx++;
    if (leaf_features[leaf_idx - 1].size() == 0) {
      return tuple<json, bool>(copy_leaf_dummy(ast), true);
    } else
      return tuple<json, bool>(ast, false);
  } else {
    return tuple<json, bool>(ast, true);
  }
}

json prune_last_jd(const vector<json> &records, const json &record2) {
  vector<counter> other_features_count;
  for (const nl::json &record : records) {
    counter tmp = list_toCounter(record["features"]);
    other_features_count.push_back(tmp);
  }
  py::object featurize = py::module_::import("featurize");
  py::object collect_features_as_list =
      featurize.attr("collect_features_as_list");
  vector<counter> leaves_features_count =
      collect_features_as_list(record2["ast"], false, true)
          .cast<vector<counter>>();
  vector<counter> out_features(leaves_features_count.size());
  counter current_features_count;
  for (counter &features1 : other_features_count) {
    double score = jaccard(features1, current_features_count);
    bool done = false;
    while (!done) {
      double max = score;
      int max_idx = -1;
      int i = 0;
      for (counter leaf_features_count : leaves_features_count) {
        if (leaf_features_count.size() > 0) {
          counter new_feautures_count =
              union_counter(current_features_count, leaf_features_count);
          double tmp = jaccard(features1, new_feautures_count);
          if (tmp > max)
            max = tmp, max_idx = i;
        }
        i++;
      }
      if (max_idx != -1) {
        score = max;
        out_features[max_idx] = leaves_features_count[max_idx];
        current_features_count = union_counter(current_features_count,
                                               leaves_features_count[max_idx]);
        leaves_features_count[max_idx].clear();
      } else
        done = true;
    }
  }
  int leaf_idx = 0;
  json prune_ast_ = get<0>(prune_ast(record2["ast"], out_features, leaf_idx));
  py::object result = collect_features_as_list(prune_ast_, false, false);
  vector<int> pruned_features = result.cast<vector<int>>();
  json res = json(record2);
  res["ast"] = prune_ast_;
  res["features"] = pruned_features;
  res["index"] = -1;
  return res;
}

vector<candidate_tuple> prune_parallel(const json &query_record,
                                       const double &min_pruned_score,
                                       const vector<json> &similar_records,
                                       const vector<double> &scores) {
  vector<candidate_tuple> candidate_records;
   //thread t[2];
  for (int i = 0; i < similar_records.size(); i++) {
    // cout << "start" << i << endl;
    //t[i % 2] = thread(
    [&](nl::json similar_record, double score) {
      // cout << "entering process" << endl;
      nl::json pruned_record =
          prune_last_jd(vector<json>(1, query_record), similar_record);
      // cout << pruned_record <<endl;
      // cout << "finished pruning" << endl;
      counter c1 = list_toCounter(query_record["features"]),
              c2 = list_toCounter(pruned_record["features"]);
      // cout << "finished counting" << endl;
      double prune_score = jaccard(c1, c2);
      // cout << prune_score << endl;
      // mutex m;
      if (prune_score > min_pruned_score) {
        // m.lock();
        candidate_records.push_back(
            candidate_tuple(similar_record, score, pruned_record, prune_score));
        // cout << "finished push_back" << endl;
        // m.unlock();
      }
    }(similar_records[i], scores[i]);
    // ,similar_records[i], scores[i]);
    //  if ((i + 1) % 2 == 0) {
    //    cout << "waiting..." << endl;
    //    for (int i = 0; i < 2; i++) {
    //      if(t[i].joinable()) t[i].join();
    //    }
    //  }
  }
  // cout << "finished." << endl;
  return candidate_records;
}

vector<clustered_record> cluster_and_intersect(
    const json &query_record, const vector<candidate_tuple> &candidate_records,
    const int &top_n, const double &threshold1, const double &threshold2) {
  size_t len_candidate = candidate_records.size();
  vector<clustered_record> clustered_records;
  if (len_candidate > 0) {
    vector<vector<int>> ret;
    vector<vector<int>> acc;
    for (int i = 0; i < len_candidate; ++i) {
      size_t cs = get<0>(candidate_records[i])["features"].size();
      size_t csq = get<2>(candidate_records[i])["features"].size();
      if (cs > csq * threshold2) {
        ret.push_back(vector<int>{i});
      }
    }
    bool changed = true;
    while (changed) {
      vector<vector<int>> tmp;
      changed = false;
      for (vector<int> &tuple : ret) {
        int kmax = -1;
        int maxscore = 0;
        vector<json> prune_record_list;
        vector<json> original_record_list;
        for (int i : tuple) {
          prune_record_list.push_back(get<2>(candidate_records[i]));
          original_record_list.push_back(get<0>(candidate_records[i]));
        }
        int qlen = (prune_record_list[0]["features"]).size();
        for (int k = tuple[tuple.size() - 1] + 1; k < len_candidate; ++k) {
          prune_record_list.push_back(get<2>(candidate_records[k]));
          original_record_list.push_back(get<0>(candidate_records[k]));
          // py::object recommand = py::module::import("recommand");
          // py::object find_similarity_score_features_set_un =
          // recommand.attr("find_similarity_score_features_set_un");
          int csq = find_similarity_score_features_set_un(
              prune_record_list); //.cast<int>();
          double pscore = 1.0 * csq / qlen;
          if (pscore > threshold1) {
            int cs = find_similarity_score_features_set_un(
                original_record_list); //.cast<int>();
            if (cs > threshold2 * csq && cs > maxscore) {
              kmax = k;
              maxscore = cs;
            }
          }
          original_record_list.pop_back();
          prune_record_list.pop_back();
        }
        if (kmax != -1) {
          changed = true;
          tuple.push_back(kmax);
          tmp.push_back(tuple);
        }
      }
      acc.insert(acc.end(), tmp.begin(), tmp.end());
      ret = tmp;
    }
    sort(acc.begin(), acc.end(), [](vector<int> &t1, vector<int> &t2) {
      return t1[0] * 1000 - t1.size() < t2[0] * 1000 - t2.size();
    });
    if (acc.size() > 0) {
      for (int i = 0; i < acc.size(); ++i) {
        vector<int> &tuple = acc[i];
        bool is_subset = false;
        for (int j = i - 1; j >= 0; j--) {
          if (jaccard(list_toCounter(tuple), list_toCounter(acc[j])) > 0.5) {
            is_subset = false;
            break;
          }
        }
        if (!is_subset) {
          json pruned_record = get<2>(candidate_records[tuple[0]]);
          for (int j = 1; j < tuple.size(); ++j) {
            pruned_record = prune_last_jd(
                vector<json>{query_record, get<0>(candidate_records[tuple[j]])},
                pruned_record);
          }
          clustered_records.push_back(clustered_record{
              pruned_record, get<0>(candidate_records[tuple[0]])});
          if (clustered_records.size() >= top_n) {
            return clustered_records;
          }
        }
      }
    }
  }
  return clustered_records;
}

PYBIND11_MODULE(cpp_module, m) {
  m.doc() = "support effective computting in recommending process!";
  m.def("jaccard", &jaccard);
  m.def("prune_last_jd", &prune_last_jd);
  m.def("prune_parallel", &prune_parallel);
  m.def("cluster_and_intersect", &cluster_and_intersect);
  m.def("find_similarity_score_features_set_un",
        &find_similarity_score_features_set_un);
}