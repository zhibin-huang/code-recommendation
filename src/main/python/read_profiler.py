import pstats
p = pstats.Stats("tmpout/profiler")
p.strip_dirs().sort_stats("cumulative", "name").print_stats(20)