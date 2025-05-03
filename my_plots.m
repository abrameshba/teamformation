reset
#set datafile separator ","
#set terminal x11 persist
set logscale y
set xrange [3:21]
set key Left left
set xlabel "Task Size"
set ylabel "Processing time(ns)"
#set title "random"
#set terminal tikz standalone
#set output "/home/ramesh/dblp/output/eps/".network."_processing_time.tex"
set term eps
set output "/home/ramesh/dblp/output/eps/".network."_processing_time.eps"
plot "/home/ramesh/dblp/output/".network."_blend_analysis.txt" using 1:2 with linespoints pt 5 ps 1 dt 2 title "Blend", \
    "/home/ramesh/dblp/output/".network."_popular_analysis.txt" using 1:2 with linespoints pt 7 ps 1 dt 2 title "Popular", \
    "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:2 with linespoints pt 9 ps 1 dt 2 title "ACO",\
    "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:2 with linespoints pt 13 ps 1 dt 2 title "NSGA-II"

reset
#set datafile separator ","
#set terminal x11 persist
#set logscale y
set xrange [3:21]
set key Left left
set xlabel "Task size"
set ylabel "Gamma Diversity"
#set title "random"
#set terminal tikz standalone
#set output "/home/ramesh/dblp/output/eps/".network."_gamma_diversity.tex"
set term eps
set output "/home/ramesh/dblp/output/eps/".network."_gamma_diversity.eps"
#plot "/home/ramesh/dblp/output/".network."_blend_analysis.txt" using 1:7 with linespoints pt 5 ps 1 dt 2 title "Blend",\
#    "/home/ramesh/dblp/output/".network."_popular_analysis.txt" using 1:7 with linespoints pt 7 ps 1 dt 2 title "Popular", \
#    "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:7 with linespoints pt 9 ps 1 dt 2 title "ACO",\
#    "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:7 with linespoints pt 13 ps 1 dt 2 title "NSGA-II"

plot "/home/ramesh/dblp/output/".network."_blend_analysis.txt" using 1:($7) with linespoints pt 5 ps 1 dt 2 title "Blend", \
    "/home/ramesh/dblp/output/".network."_popular_analysis.txt" using 1:($7) with linespoints pt 7 ps 1 dt 2 title "Popular", \
    "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($7) with linespoints pt 9 ps 1 dt 2 title "ACO", \
    "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($7) with linespoints pt 13 ps 1 dt 2 title "NSGA-II"

# CC-diameter
reset
#set datafile separator ","
#set terminal x11 persist
#set logscale y
set xrange [3:21]
set key Left left
set xlabel "Task size"
set ylabel "Gamma Diversity/Diameter"
#set title "random"
#set terminal tikz standalone
#set output "/home/ramesh/dblp/output/eps/".network."_gamma_diversity_diameter.tex"
set term eps
set output "/home/ramesh/dblp/output/eps/".network."_gamma_diversity_diameter.eps"
#plot "/home/ramesh/dblp/output/".network."_blend_analysis.txt" using 1:($7/$4) with linespoints pt 5 ps 1 dt 2 title "Blend",\
#    "/home/ramesh/dblp/output/".network."_popular_analysis.txt" using 1:($7/$4) with linespoints pt 7 ps 1 dt 2 title "Popular", \
#    "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($7/$4) with linespoints pt 9 ps 1 dt 2 title "ACO",\
#    "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($7/$4) with linespoints pt 13 ps 1 dt 2 title "NSGA-II"

plot "/home/ramesh/dblp/output/".network."_blend_analysis.txt" using 1:($7/$4) with linespoints pt 5 ps 1 dt 2 title "Blend", \
    "/home/ramesh/dblp/output/".network."_popular_analysis.txt" using 1:($7/$4) with linespoints pt 7 ps 1 dt 2 title "Popular", \
    "/home/ramesh/dblp/output/".network."_aco_analysis.txt" using 1:($7/$4) with linespoints pt 9 ps 1 dt 2 title "ACO", \
    "/home/ramesh/dblp/output/".network."_nsga2_analysis.txt" using 1:($7/$4) with linespoints pt 13 ps 1 dt 2 title "NSGA-II"