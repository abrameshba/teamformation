#!/usr/bin/gnuplot -persist

reset
set terminal x11 persist
set key Left left
set boxwidth 0.75 relative
set style data histograms
set datafile separator comma
set style histogram rowstacked
set key autotitle columnhead
set style fill solid 1.0 border lt -1
set xtics rotate
set xlabel "Year"
set ylabel "#Records"
set ytics add ('100K' 100000, '200K' 200000, '300K' 300000, '400K' 400000, '500K' 500000, '600K' 600000)
set for [i=0:100000:600000] ytics
set terminal postscript eps enhanced color
set output '/home/ramesh/dblp/output/eps/publications.eps'
plot  for [COL=2:7:1] '/home/ramesh/dblp/publications-per-year-from-1990.csv'  using COL:xticlabels(1)
