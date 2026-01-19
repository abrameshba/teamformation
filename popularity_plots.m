#!/usr/bin/gnuplot -persist
do for [network in "icdt colt pods pkdd ecml sdm stacs uai edbt stoc soda focs icml icdm vldb www kdd sigmod icde ai th db dm dblp"] {
reset
set terminal postscript eps enhanced color font 'Arial-Bold'
#set title "Experts per skill" icdt_popularity_summary
stats '/home/ramesh/dblp/output/'.network.'_popularity_summary.txt' using 1:2 nooutput
avg=STATS_sumxy/STATS_sum_y
set xrange [STATS_min_x/2:STATS_max_x*2]
set yrange [STATS_min_y/2:STATS_max_y*2]
set xlabel "Popularity"
set ylabel "Number of skills"
set logscale xy
set arrow from avg,STATS_min_y to avg,STATS_max_y heads dt "."
set arrow from avg,STATS_max_y/2 to avg*2,STATS_max_y/2
set arrow from avg,STATS_min_y*2 to avg/2,STATS_min_y*2
set label "Popular skills" at avg*2,STATS_max_y/2
set label "Rare skills" at avg/6,STATS_min_y*2
set label sprintf("avg-pop = %3.2f",avg) at avg/2,STATS_max_y+15
set output '/home/ramesh/dblp/output/eps/'.network.'-popularity-pl.eps'
a=10
b=.10
fn(x) = a*x**(-b)
fit fn(x) '/home/ramesh/dblp/output/'.network.'_popularity_summary.txt' via a, b
plot    '/home/ramesh/dblp/output/'.network.'_popularity_summary.txt' using (($1>0)? $1 : 1/0):2  with point pointtype 3 pointsize 2 lc rgb "#0000FF" title "Popularity of a skill", fn(x) title "popularity"  lt 2 lw 1
#print(network)
#print(a)
#print(b)
}