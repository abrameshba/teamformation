#!/usr/bin/gnuplot -persist

# Following code is to draw plots between task size and processing time


do for[network in "vldb"] {
reset
#set datafile separator ","
#set terminal x11 persist
set logscale y
# set yrange [10000000:10000000000000]
set xrange [3:21]
set key Left left
set xlabel 'Task size'
set ylabel 'Processing time (ns)'
set key font "Courier New,10"
#set terminal tikz standalone
#set output "/home/ramesh/dblp/output/eps/".network."_process_time.tex"
#plot "/home/ramesh/dblp/output/".network."_blend_analysis.txt" using 1:2 with linespoints pt 5 ps 1.2 dt 2 title "Blend",\
#    "/home/ramesh/dblp/output/".network."_popular_analysis.txt" using 1:2 with linespoints pt 7 ps 1.2 dt 2 title "Popular", \
#    "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:2 with linespoints pt 9 ps 1 dt 2 title "ACO",\
#    "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:2 with linespoints pt 13 ps 1 dt 2 title "NSGA-II"
set terminal eps color
set output "/home/ramesh/dblp/output/eps/".network."_process_time.eps"
plot "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:2 with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output/".network."_blend_analysis.txt" using 1:2 with linespoints pt 5 ps 1 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output/".network."_cmopso_analysis.txt" using 1:2 with linespoints pt 3 ps 1 dt 2 lc "brown"  title "CMOPSO",\
    "/home/ramesh/dblp/output/".network."_hybrid_analysis.txt" using 1:2 with linespoints pt 11 ps 1 dt 2 lc "black"  title "Hybrid",\
     "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:2 with linespoints pt 15 ps 1 dt 2 lc "dark-yellow" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_nsga3_analysis.txt" using 1:2 with linespoints pt 13 ps 1 dt 2 lc "violet" title "NSGA-III", \
    "/home/ramesh/dblp/output/".network."_popular_analysis.txt" using 1:2 with linespoints pt 7 ps 1 dt 2 lc "green" title "Popular"
}
# Gamma diversity

do for[network in "vldb"] {
reset
#set datafile separator ","
#set terminal x11 persist
#set logscale y
# set yrange [0:27]
set xrange [3:21]
#stats '/home/ramesh/dblp/output/'.network.'_popularity_summary.txt' using 1:($7/$4) nooutput
#set yrange [STATS_min_y/2:STATS_max_y*2]
set key Left left
set xlabel 'Task size'
set ylabel 'Gamma/Diameter'
set terminal eps color
set key font "Courier New,10"
set output "/home/ramesh/dblp/output/eps/".network."_gamma_diameter.eps"
plot "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($7/$4) with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output/".network."_blend_analysis.txt" using 1:($7/$4) with linespoints pt 5 ps 1 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output/".network."_cmopso_analysis.txt" using 1:($7/$4) with linespoints pt 3 ps 1 dt 2 lc "brown"  title "CMOPSO",\
    "/home/ramesh/dblp/output/".network."_hybrid_analysis.txt" using 1:($7/$4) with linespoints pt 11 ps 1 dt 2 lc "black"  title "Hybrid",\
     "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($7/$4) with linespoints pt 15 ps 1 dt 2 lc "dark-yellow" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_nsga3_analysis.txt" using 1:($7/$4) with linespoints pt 13 ps 1 dt 2 lc "violet" title "NSGA-III", \
    "/home/ramesh/dblp/output/".network."_popular_analysis.txt" using 1:($7/$4) with linespoints pt 7 ps 1 dt 2 lc "green" title "Popular"
}


do for[network in "vldb"] {
reset
#set datafile separator ","
#set terminal x11 persist
#set logscale y
# set yrange [0:9]
set xrange [3:21]
#stats '/home/ramesh/dblp/output/'.network.'_popularity_summary.txt' using 1:($7/$5) nooutput
#set yrange [STATS_min_y/2:STATS_max_y*2]
set key Left right
set xlabel 'Task size'
set ylabel 'Gamma/LeaderDistance'
set terminal eps color
set key font "Courier New,10"
set output "/home/ramesh/dblp/output/eps/".network."_gamma_ld.eps"
plot "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($7/$5) with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output/".network."_blend_analysis.txt" using 1:($7/$5) with linespoints pt 5 ps 1 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output/".network."_cmopso_analysis.txt" using 1:($7/$5) with linespoints pt 3 ps 1 dt 2 lc "brown"  title "CMOPSO",\
    "/home/ramesh/dblp/output/".network."_hybrid_analysis.txt" using 1:($7/$5) with linespoints pt 11 ps 1 dt 2 lc "black"  title "Hybrid",\
     "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($7/$5) with linespoints pt 15 ps 1 dt 2 lc "dark-yellow" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_nsga3_analysis.txt" using 1:($7/$5) with linespoints pt 13 ps 1 dt 2 lc "violet" title "NSGA-III", \
    "/home/ramesh/dblp/output/".network."_popular_analysis.txt" using 1:($7/$5) with linespoints pt 7 ps 1 dt 2 lc "green" title "Popular"
}


do for[network in "vldb"] {
reset
#set datafile separator ","
#set terminal x11 persist
#set logscale y
set xrange [3:21]
#stats '/home/ramesh/dblp/output/'.network.'_popularity_summary.txt' using 1:($7/$6) nooutput
#set yrange [STATS_min_y/2:STATS_max_y*2]
set key Left right
set xlabel 'Task size'
set ylabel 'Gamma/SumDistance'
set terminal eps color
set key font "Courier New,10"
set output "/home/ramesh/dblp/output/eps/".network."_gamma_sd.eps"
plot "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($7/$6) with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output/".network."_blend_analysis.txt" using 1:($7/$6) with linespoints pt 5 ps 1 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output/".network."_cmopso_analysis.txt" using 1:($7/$6) with linespoints pt 3 ps 1 dt 2 lc "brown"  title "CMOPSO",\
    "/home/ramesh/dblp/output/".network."_hybrid_analysis.txt" using 1:($7/$6) with linespoints pt 11 ps 1 dt 2 lc "black"  title "Hybrid",\
     "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($7/$6) with linespoints pt 15 ps 1 dt 2 lc "dark-yellow" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_nsga3_analysis.txt" using 1:($7/$6) with linespoints pt 13 ps 1 dt 2 lc "violet" title "NSGA-III", \
    "/home/ramesh/dblp/output/".network."_popular_analysis.txt" using 1:($7/$6) with linespoints pt 7 ps 1 dt 2 lc "green" title "Popular"
}
# Shannon diversity

do for[network in "vldb"] {
reset
#set datafile separator ","
#set terminal x11 persist
#set logscale y
# set yrange [0:3]
set xrange [3:21]
#stats '/home/ramesh/dblp/output/'.network.'_popularity_summary.txt' using 1:($8/$4) nooutput
#set yrange [STATS_min_y/2:STATS_max_y*2]
set key Left left
set xlabel 'Task size'
set ylabel 'Shannon/Diameter'
set terminal eps color
set key font "Courier New,10"
set output "/home/ramesh/dblp/output/eps/".network."_shannon_diameter.eps"
plot "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($8/$4) with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output/".network."_blend_analysis.txt" using 1:($8/$4) with linespoints pt 5 ps 1 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output/".network."_cmopso_analysis.txt" using 1:($8/$4) with linespoints pt 3 ps 1 dt 2 lc "brown"  title "CMOPSO",\
    "/home/ramesh/dblp/output/".network."_hybrid_analysis.txt" using 1:($8/$4) with linespoints pt 11 ps 1 dt 2 lc "black"  title "Hybrid",\
     "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($8/$4) with linespoints pt 15 ps 1 dt 2 lc "dark-yellow" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_nsga3_analysis.txt" using 1:($8/$4) with linespoints pt 13 ps 1 dt 2 lc "violet" title "NSGA-III", \
    "/home/ramesh/dblp/output/".network."_popular_analysis.txt" using 1:($8/$4) with linespoints pt 7 ps 1 dt 2 lc "green" title "Popular"
}


do for[network in "vldb"] {
reset
#set datafile separator ","
#set terminal x11 persist
#set logscale y
# set yrange [0:1]
set xrange [3:21]
#stats '/home/ramesh/dblp/output/'.network.'_popularity_summary.txt' using 1:($8/$5) nooutput
#set yrange [STATS_min_y/2:STATS_max_y*2]
set key Left right
set xlabel 'Task size'
set ylabel 'Shannon/LeaderDistance'
set terminal eps color
set key font "Courier New,10"
set output "/home/ramesh/dblp/output/eps/".network."_shannon_ld.eps"
plot "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($8/$5) with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output/".network."_blend_analysis.txt" using 1:($8/$5) with linespoints pt 5 ps 1 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output/".network."_cmopso_analysis.txt" using 1:($8/$5) with linespoints pt 3 ps 1 dt 2 lc "brown"  title "CMOPSO",\
    "/home/ramesh/dblp/output/".network."_hybrid_analysis.txt" using 1:($8/$5) with linespoints pt 11 ps 1 dt 2 lc "black"  title "Hybrid",\
     "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($8/$5) with linespoints pt 15 ps 1 dt 2 lc "dark-yellow" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_nsga3_analysis.txt" using 1:($8/$5) with linespoints pt 13 ps 1 dt 2 lc "violet" title "NSGA-III", \
    "/home/ramesh/dblp/output/".network."_popular_analysis.txt" using 1:($8/$5) with linespoints pt 7 ps 1 dt 2 lc "green" title "Popular"
}

do for[network in "vldb"] {
reset
#set datafile separator ","
#set terminal x11 persist
#set logscale y
# set yrange [0:.4]
set xrange [3:21]
#stats '/home/ramesh/dblp/output/'.network.'_popularity_summary.txt' using 1:($8/$6) nooutput
#set yrange [STATS_min_y/2:STATS_max_y*2]
set key Left right
set xlabel 'Task size'
set ylabel 'Shannon/SumDistance'
set terminal eps color
set key font "Courier New,10"
set output "/home/ramesh/dblp/output/eps/".network."_shannon_sd.eps"
plot "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($8/$6) with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output/".network."_blend_analysis.txt" using 1:($8/$6) with linespoints pt 5 ps 1 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output/".network."_cmopso_analysis.txt" using 1:($8/$6) with linespoints pt 3 ps 1 dt 2 lc "brown"  title "CMOPSO",\
    "/home/ramesh/dblp/output/".network."_hybrid_analysis.txt" using 1:($8/$6) with linespoints pt 11 ps 1 dt 2 lc "black"  title "Hybrid",\
     "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($8/$6) with linespoints pt 15 ps 1 dt 2 lc "dark-yellow" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_nsga3_analysis.txt" using 1:($8/$6) with linespoints pt 13 ps 1 dt 2 lc "violet" title "NSGA-III", \
    "/home/ramesh/dblp/output/".network."_popular_analysis.txt" using 1:($8/$6) with linespoints pt 7 ps 1 dt 2 lc "green" title "Popular"
}

# Inverse simpson diversity


do for[network in "vldb"] {
reset
#set datafile separator ","
#set terminal x11 persist
#set logscale y
# set yrange [0.02:.12]
set xrange [3:21]
#stats '/home/ramesh/dblp/output/'.network.'_popularity_summary.txt' using 1:($9/$4) nooutput
#set yrange [STATS_min_y/2:STATS_max_y*2]
set key Left left
set xlabel 'Task size'
set ylabel 'Inverse Simpson/Diameter'
set terminal eps color
set key font "Courier New,10"
set output "/home/ramesh/dblp/output/eps/".network."_isn_diameter.eps"
plot "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($9/$4) with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output/".network."_blend_analysis.txt" using 1:($9/$4) with linespoints pt 5 ps 1 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output/".network."_cmopso_analysis.txt" using 1:($9/$4) with linespoints pt 3 ps 1 dt 2 lc "brown"  title "CMOPSO",\
    "/home/ramesh/dblp/output/".network."_hybrid_analysis.txt" using 1:($9/$4) with linespoints pt 11 ps 1 dt 2 lc "black"  title "Hybrid",\
     "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($9/$4) with linespoints pt 15 ps 1 dt 2 lc "dark-yellow" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_nsga3_analysis.txt" using 1:($9/$4) with linespoints pt 13 ps 1 dt 2 lc "violet" title "NSGA-III", \
    "/home/ramesh/dblp/output/".network."_popular_analysis.txt" using 1:($9/$4) with linespoints pt 7 ps 1 dt 2 lc "green" title "Popular"
}


do for[network in "vldb"] {
reset
#set datafile separator ","
#set terminal x11 persist
#set logscale y
# set yrange [0:0.035]
set xrange [3:21]
#stats '/home/ramesh/dblp/output/'.network.'_popularity_summary.txt' using 1:($9/$5) nooutput
#set yrange [STATS_min_y/2:STATS_max_y*2]
set key Left right
set xlabel 'Task size'
set ylabel 'Inverse Simpson/LeaderDistance'
set terminal eps color
set key font "Courier New,10"
set output "/home/ramesh/dblp/output/eps/".network."_isn_ld.eps"
plot "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($9/$5) with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output/".network."_blend_analysis.txt" using 1:($9/$5) with linespoints pt 5 ps 1 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output/".network."_cmopso_analysis.txt" using 1:($9/$5) with linespoints pt 3 ps 1 dt 2 lc "brown"  title "CMOPSO",\
    "/home/ramesh/dblp/output/".network."_hybrid_analysis.txt" using 1:($9/$5) with linespoints pt 11 ps 1 dt 2 lc "black"  title "Hybrid",\
     "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($9/$5) with linespoints pt 15 ps 1 dt 2 lc "dark-yellow" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_nsga3_analysis.txt" using 1:($9/$5) with linespoints pt 13 ps 1 dt 2 lc "violet" title "NSGA-III", \
    "/home/ramesh/dblp/output/".network."_popular_analysis.txt" using 1:($9/$5) with linespoints pt 7 ps 1 dt 2 lc "green" title "Popular"
}

do for[network in "vldb"] {
reset
#set datafile separator ","
#set terminal x11 persist
#set logscale y
# set yrange [0:0.017]
set xrange [3:21]
#stats '/home/ramesh/dblp/output/'.network.'_popularity_summary.txt' using 1:($9/$6) nooutput
#set yrange [STATS_min_y/2:STATS_max_y*2]
set key Left right
set xlabel 'Task size'
set ylabel 'Inverse Simpson/SumDistance'
set terminal eps color
set key font "Courier New,10"
set output "/home/ramesh/dblp/output/eps/".network."_isn_sd.eps"
plot "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($9/$6) with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output/".network."_blend_analysis.txt" using 1:($9/$6) with linespoints pt 5 ps 1 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output/".network."_cmopso_analysis.txt" using 1:($9/$6) with linespoints pt 3 ps 1 dt 2 lc "brown"  title "CMOPSO",\
    "/home/ramesh/dblp/output/".network."_hybrid_analysis.txt" using 1:($9/$6) with linespoints pt 11 ps 1 dt 2 lc "black"  title "Hybrid",\
     "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($9/$6) with linespoints pt 15 ps 1 dt 2 lc "dark-yellow" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_nsga3_analysis.txt" using 1:($9/$6) with linespoints pt 13 ps 1 dt 2 lc "violet" title "NSGA-III", \
    "/home/ramesh/dblp/output/".network."_popular_analysis.txt" using 1:($9/$6) with linespoints pt 7 ps 1 dt 2 lc "green" title "Popular"
}