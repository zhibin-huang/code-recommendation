# installation
mvn clean package

# compile entire corpus to ast json file and create jsrc.json from all Java files under jsrc
time mvn exec:java -Dexec.mainClass=ConvertJava -Dexec.args="compilationUnit /Users/huangzhibin/Desktop/jsrc.json /Users/huangzhibin/Desktop/java1000"
# convert query_file.java into ast in json format
time mvn exec:java -Dexec.mainClass=ConvertJava -Dexec.args="compilationUnit example_data/example_query.json example_data/example_query.java"

# featurize corpus
time python3 src/main/python/entry.py -c ./repos -d ./tmpout
# run experiments assuming that featurization has already been done
time python3 -m cProfile -o tmpout/profiler src/main/python/entry.py -d ./tmpout -t
# search code at index 83403 of the corpus
time python3 src/main/python/entry.py -d ./tmpout -i 83403
# search using the query ast in query_file.json
time python3 src/main/python/entry.py -d ./tmpout -f query_file.json
# compile cpp module
src/main/python/compile.sh src/main/python/cpp_module
# read profile
python3 src/main/python/read_profiler.py
# start service
python3  src/main/python/entry.py -d ./tmpout
