#!/usr/bin/gnuplot -persist

# Following code is to draw plots between task size and processing time

do for [network in "icdt colt pods pkdd ecml sdm stacs uai edbt stoc soda focs icml icdm vldb kdd sigmod icde ai"] {

reset
#set datafile separator ","
#set terminal x11 persist
set logscale y
set xrange [3:21]
set key Left left
set xlabel 'Task size'
set ylabel 'Processing time (ns)'
#set title "random"
#set terminal tikz standalone
#set output "/home/ramesh/dblp/output_sdb/eps/".network."_process_time.tex"
#plot "/home/ramesh/dblp/output_sdb/".network."_blend_analysis.txt" using 1:2 with linespoints pt 5 ps 1.2 dt 2 title "Blend",\
#    "/home/ramesh/dblp/output_sdb/".network."_popular_analysis.txt" using 1:2 with linespoints pt 7 ps 1.2 dt 2 title "Popular", \
#    "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:2 with linespoints pt 9 ps 1 dt 2 title "ACO",\
#    "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:2 with linespoints pt 13 ps 1 dt 2 title "NSGA-II"
set terminal eps color
set output "/home/ramesh/dblp/output_sdb/eps/".network."_process_time.eps"
plot "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:2 with linespoints pt 13 ps 1 dt 2 lc "black" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:2 with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output_sdb/".network."_blend_analysis.txt" using 1:2 with linespoints pt 5 ps 1.2 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output_sdb/".network."_popular_analysis.txt" using 1:2 with linespoints pt 7 ps 1.2 dt 2 lc "green" title "Popular"
}

do for [network in "icdt colt pods pkdd ecml sdm stacs uai edbt stoc soda focs icml icdm vldb kdd sigmod icde ai"] {

reset
#set datafile separator ","
#set terminal x11 persist
set logscale y
set xrange [3:21]
#stats '/home/ramesh/dblp/output/'.network.'_popularity_summary.txt' using 1:($7/$4) nooutput
#set yrange [STATS_min_y/2:STATS_max_y*2]
set key Left left
set xlabel 'Task size'
set ylabel 'Gamma/Diameter'
#set title "random"
set terminal eps color
set output "/home/ramesh/dblp/output_sdb/eps/".network."_gamma_diameter_log.eps"
plot "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($7/$4) with linespoints pt 13 ps 1 dt 2 lc "black" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($7/$4) with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output_sdb/".network."_blend_analysis.txt" using 1:($7/$4) with linespoints pt 5 ps 1.2 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output_sdb/".network."_popular_analysis.txt" using 1:($7/$4) with linespoints pt 7 ps 1.2 dt 2 lc "green" title "Popular"
}

do for [network in "icdt colt pods pkdd ecml sdm stacs uai edbt stoc soda focs icml icdm vldb kdd sigmod icde ai"] {

reset
#set datafile separator ","
#set terminal x11 persist
set xrange [3:21]
set key Left left
set xlabel 'Task size'
set ylabel 'Gamma/Diameter'
#set title "random"
set terminal eps color
set output "/home/ramesh/dblp/output_sdb/eps/".network."_gamma_diameter.eps"
plot "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($7/$4) with linespoints pt 13 ps 1 dt 2 lc "black" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($7/$4) with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output_sdb/".network."_blend_analysis.txt" using 1:($7/$4) with linespoints pt 5 ps 1.2 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output_sdb/".network."_popular_analysis.txt" using 1:($7/$4) with linespoints pt 7 ps 1.2 dt 2 lc "green" title "Popular"
}

do for [network in "icdt colt pods pkdd ecml sdm stacs uai edbt stoc soda focs icml icdm vldb kdd sigmod icde ai"] {

reset
#set datafile separator ","
#set terminal x11 persist
set logscale y
set xrange [3:21]
set key Left left
set xlabel 'Task size'
set ylabel 'Shannon/Diameter'
#set title "random"
set terminal eps color
set output "/home/ramesh/dblp/output_sdb/eps/".network."_shannon_diameter_log.eps"
plot "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($8/$4) with linespoints pt 13 ps 1 dt 2 lc "black" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($8/$4) with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output_sdb/".network."_blend_analysis.txt" using 1:($8/$4) with linespoints pt 5 ps 1.2 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output_sdb/".network."_popular_analysis.txt" using 1:($8/$4) with linespoints pt 7 ps 1.2 dt 2 lc "green" title "Popular"
}

do for [network in "icdt colt pods pkdd ecml sdm stacs uai edbt stoc soda focs icml icdm vldb kdd sigmod icde ai"] {

reset
#set datafile separator ","
#set terminal x11 persist
set xrange [3:21]
set key Left left
set xlabel 'Task size'
set ylabel 'Shannon/Diameter'
#set title "random"
set terminal eps color
set output "/home/ramesh/dblp/output_sdb/eps/".network."_shannon_diameter.eps"
plot "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($8/$4) with linespoints pt 13 ps 1 dt 2 lc "black" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($8/$4) with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output_sdb/".network."_blend_analysis.txt" using 1:($8/$4) with linespoints pt 5 ps 1.2 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output_sdb/".network."_popular_analysis.txt" using 1:($8/$4) with linespoints pt 7 ps 1.2 dt 2 lc "green" title "Popular"
}

do for [network in "icdt colt pods pkdd ecml sdm stacs uai edbt stoc soda focs icml icdm vldb kdd sigmod icde ai"] {

reset
#set datafile separator ","
#set terminal x11 persist
set logscale y
set xrange [3:21]
set key Left left
set xlabel 'Task size'
set ylabel 'Inverse Gini-Simpson/Diameter'
#set title "random"
set terminal eps color
set output "/home/ramesh/dblp/output_sdb/eps/".network."_igs_diameter_log.eps"
plot "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($9/$4) with linespoints pt 13 ps 1 dt 2 lc "black" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($9/$4) with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output_sdb/".network."_blend_analysis.txt" using 1:($9/$4) with linespoints pt 5 ps 1.2 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output_sdb/".network."_popular_analysis.txt" using 1:($9/$4) with linespoints pt 7 ps 1.2 dt 2 lc "green" title "Popular"
}

do for [network in "icdt colt pods pkdd ecml sdm stacs uai edbt stoc soda focs icml icdm vldb kdd sigmod icde ai"] {

reset
#set datafile separator ","
#set terminal x11 persist
set xrange [3:21]
set key Left left
set xlabel 'Task size'
set ylabel 'Inverse Gini-Simpson/Diameter'
#set title "random"
set terminal eps color
set output "/home/ramesh/dblp/output_sdb/eps/".network."_igs_diameter.eps"
plot "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($9/$4) with linespoints pt 13 ps 1 dt 2 lc "black" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($9/$4) with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output_sdb/".network."_blend_analysis.txt" using 1:($9/$4) with linespoints pt 5 ps 1.2 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output_sdb/".network."_popular_analysis.txt" using 1:($9/$4) with linespoints pt 7 ps 1.2 dt 2 lc "green" title "Popular"
}

do for [network in "icdt colt pods pkdd ecml sdm stacs uai edbt stoc soda focs icml icdm vldb kdd sigmod icde ai"] {

reset
#set datafile separator ","
#set terminal x11 persist
set logscale y
set xrange [3:21]
set key Left right
set xlabel 'Task size'
set ylabel 'Gamma/SumDistance'
#set title "random"
set terminal eps color
set output "/home/ramesh/dblp/output_sdb/eps/".network."_gamma_sumdstnc_log.eps"
plot "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($7/$6) with linespoints pt 13 ps 1 dt 2 lc "black" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($7/$6) with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output_sdb/".network."_blend_analysis.txt" using 1:($7/$6) with linespoints pt 5 ps 1.2 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output_sdb/".network."_popular_analysis.txt" using 1:($7/$6) with linespoints pt 7 ps 1.2 dt 2 lc "green" title "Popular"
}

do for [network in "icdt colt pods pkdd ecml sdm stacs uai edbt stoc soda focs icml icdm vldb kdd sigmod icde ai"] {

reset
#set datafile separator ","
#set terminal x11 persist
set xrange [3:21]
set key Left right
set xlabel 'Task size'
set ylabel 'Gamma/SumDistance'
#set title "random"
set terminal eps color
set output "/home/ramesh/dblp/output_sdb/eps/".network."_gamma_sumdstnc.eps"
plot "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($7/$6) with linespoints pt 13 ps 1 dt 2 lc "black" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($7/$6) with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output_sdb/".network."_blend_analysis.txt" using 1:($7/$6) with linespoints pt 5 ps 1.2 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output_sdb/".network."_popular_analysis.txt" using 1:($7/$6) with linespoints pt 7 ps 1.2 dt 2 lc "green" title "Popular"
}

do for [network in "icdt colt pods pkdd ecml sdm stacs uai edbt stoc soda focs icml icdm vldb kdd sigmod icde ai"] {

reset
#set datafile separator ","
#set terminal x11 persist
set logscale y
set xrange [3:21]
set key Left right
set xlabel 'Task size'
set ylabel 'Shannon/SumDistance'
#set title "random"
set terminal eps color
set output "/home/ramesh/dblp/output_sdb/eps/".network."_shannon_sumdstnc_log.eps"
plot "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($8/$6) with linespoints pt 13 ps 1 dt 2 lc "black" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($8/$6) with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output_sdb/".network."_blend_analysis.txt" using 1:($8/$6) with linespoints pt 5 ps 1.2 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output_sdb/".network."_popular_analysis.txt" using 1:($8/$6) with linespoints pt 7 ps 1.2 dt 2 lc "green" title "Popular"
}

do for [network in "icdt colt pods pkdd ecml sdm stacs uai edbt stoc soda focs icml icdm vldb kdd sigmod icde ai"] {

reset
#set datafile separator ","
#set terminal x11 persist
set xrange [3:21]
set key Left right
set xlabel 'Task size'
set ylabel 'Shannon/SumDistance'
#set title "random"
set terminal eps color
set output "/home/ramesh/dblp/output_sdb/eps/".network."_shannon_sumdstnc.eps"
plot "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($8/$6) with linespoints pt 13 ps 1 dt 2 lc "black" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($8/$6) with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output_sdb/".network."_blend_analysis.txt" using 1:($8/$6) with linespoints pt 5 ps 1.2 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output_sdb/".network."_popular_analysis.txt" using 1:($8/$6) with linespoints pt 7 ps 1.2 dt 2 lc "green" title "Popular"
}

do for [network in "icdt colt pods pkdd ecml sdm stacs uai edbt stoc soda focs icml icdm vldb kdd sigmod icde ai"] {

reset
#set datafile separator ","
#set terminal x11 persist
set logscale y
set xrange [3:21]
set key Left right
set xlabel 'Task size'
set ylabel 'Inverse Gini-Simpson/SumDistance'
#set title "random"
set terminal eps color
set output "/home/ramesh/dblp/output_sdb/eps/".network."_igs_sumdstnc_log.eps"
plot "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($9/$6) with linespoints pt 13 ps 1 dt 2 lc "black" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($9/$6) with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output_sdb/".network."_blend_analysis.txt" using 1:($9/$6) with linespoints pt 5 ps 1.2 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output_sdb/".network."_popular_analysis.txt" using 1:($9/$6) with linespoints pt 7 ps 1.2 dt 2 lc "green" title "Popular"
}

do for [network in "icdt colt pods pkdd ecml sdm stacs uai edbt stoc soda focs icml icdm vldb kdd sigmod icde ai"] {

reset
#set datafile separator ","
#set terminal x11 persist
set xrange [3:21]
set key Left right
set xlabel 'Task size'
set ylabel 'Inverse Gini-Simpson/SumDistance'
#set title "random"
set terminal eps color
set output "/home/ramesh/dblp/output_sdb/eps/".network."_igs_sumdstnc.eps"
plot "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($9/$6) with linespoints pt 13 ps 1 dt 2 lc "black" title "NSGA-II", \
    "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($9/$6) with linespoints pt 9 ps 1 dt 2 lc "blue" title "ACO", \
    "/home/ramesh/dblp/output_sdb/".network."_blend_analysis.txt" using 1:($9/$6) with linespoints pt 5 ps 1.2 dt 2 lc "red" title "Blend", \
    "/home/ramesh/dblp/output_sdb/".network."_popular_analysis.txt" using 1:($9/$6) with linespoints pt 7 ps 1.2 dt 2 lc "green" title "Popular"
}