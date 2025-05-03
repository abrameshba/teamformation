#!/usr/bin/gnuplot -persist

reset
#set datafile separator ","
#set terminal x11 persist
#set logscale y
network="ai"
set xrange [3:21]
#stats '/home/ramesh/dblp/output/'.network.'_popularity_summary.txt' using 1:($9/$4) nooutput
#set yrange [STATS_min_y/2:STATS_max_y*2]
set yrange [.02:.1]
#set ytics add ('2' 2, '3' 3, '4' 4, '5' 5, '6' 6, '7' 7, '8' 8)
set key Left left
set xlabel 'Task size'
set ylabel 'Inverse Gini-Simpson Divesity/Diameter'
#set title "random"
set terminal eps color
set output "/home/ramesh/dblp/output/eps/".network."_igs_diameter_log-.eps"
plot "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($9/$4) with linespoints pt 13 ps 1 dt 2 lc "black" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($9/$4) with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output/".network."_blend_analysis.txt" using 1:($9/$4) with linespoints pt 5 ps 1.1 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output/".network."_popular_analysis.txt" using 1:($9/$4) with linespoints pt 7 ps 1.1 dt 2 lc "green" title "Popular"
 # icdt yrange [2:7], ai yrange [3:18] gamma
 # icdt yrange [.25:.65] ai yrange[.4:1.8] Shannon
 # icdt yrange [.06:.24] ai yrange [.02:1] IGS