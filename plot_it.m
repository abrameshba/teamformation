#!/usr/bin/gnuplot -persist

reset
#set datafile separator ","
#set terminal x11 persist
#set logscale y
#set xrange [3:11]
set key Left left
set xlabel "Task Size"
set ylabel "Gamma(\gamma) diversity"
#set title "random"
set terminal tikz standalone
set output "/home/ramesh/dblp/output/eps/icdt_processing_time.tex"
#set term eps
#set output "/home/ramesh/dblp/output/eps/icdt_processing_time.eps"
plot "/home/ramesh/dblp/output/icdt_blend_analysis.txt" using 1:6 with points pt 2 ps 2 title "Blend", "/home/ramesh/dblp/output/icdt_rarestfirst_analysis.txt" using 1:6 with points pt 3 ps 2 title "Rarestfirst"
