#!/usr/bin/gnuplot -persist
year ="2015"
do for [network in "icdt colt pods pkdd ecml sdm stacs uai edbt stoc soda focs icml icdm vldb www kdd sigmod icde ai th db dm dblp"] {
reset
set terminal postscript eps enhanced color font 'Arial-Bold'
#set title "power law property by degree of experts of ".network." network"  tc "royalblue"
stats '/home/ramesh/dblp/output/'.network.'_nodes.txt' u 1:2 nooutput
set xrange [STATS_min_x/2:STATS_max_x*2]
set yrange [STATS_min_y/2:STATS_max_y*2]
set xlabel "Degree of experts"
set ylabel "Number of experts"
set logscale y
set logscale x
set output '/home/ramesh/dblp/output/eps/'.network.'-pl.eps'
a=100
b=-.10
fn(x) = a*x**(-b)
#set fit quiet
fit fn(x) '/home/ramesh/dblp/output/'.network.'_nodes.txt' via a, b
plot    '/home/ramesh/dblp/output/'.network.'_nodes.txt' using 1:2 with point pointtype 3 pointsize 2 title "Degree of a node", fn(x) title "Degree power law"  lt 2 lw 1
}