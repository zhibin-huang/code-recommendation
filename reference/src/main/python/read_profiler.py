import pstats
p = pstats.Stats("/Users/huangzhibin/Downloads/aroma-paper-artifacts-master/reference/dataset/tmpout1000/profiler")
p.strip_dirs().sort_stats("cumulative", "name").print_stats(20)