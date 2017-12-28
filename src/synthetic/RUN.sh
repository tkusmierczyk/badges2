#!/bin/sh
set -e

#git pull #!

NUSERS=1000
NCPUS=28
ID=07replica2
DIR="../../data/simulation_run$ID"
REPEAT=40

mkdir $DIR

python simulation.py -c $NCPUS -p mode=trends,N=$NUSERS,numtrials=$REPEAT,TF=0.5  -o $DIR/simulation_N${NUSERS}_trends
python simulation.py -c $NCPUS -p mode=influence_frac,N=$NUSERS,numtrials=$REPEAT,TF=0.5  -o $DIR/simulation_N${NUSERS}_influence_frac
python simulation.py -c $NCPUS -p mode=proportion,N=$NUSERS,numtrials=$REPEAT,TF=0.5  -o $DIR/simulation_N${NUSERS}_proportion
python simulation.py -c $NCPUS -p mode=test_influence_frac,N=$NUSERS,numtrials=$REPEAT,TF=0.5 -o $DIR/simulation_N${NUSERS}_test_influence_frac

python simulation.py -c $NCPUS -p mode=101,N=$NUSERS,trend=0,numtrials=$REPEAT,TF=0.5  -o $DIR/simulation_N${NUSERS}_mode101
python simulation.py -c $NCPUS -p mode=102,N=$NUSERS,trend=0,numtrials=$REPEAT,TF=0.5  -o $DIR/simulation_N${NUSERS}_mode102


exit 0
###############################################
#visualisation:
python simulation_plot_trends.py -i $DIR/simulation_N${NUSERS}_proportion.json -t TF
python simulation_plot_trends.py -i $DIR/simulation_N${NUSERS}_influence_frac.json -t F
python simulation_plot_trends.py -i $DIR/simulation_N${NUSERS}_test_influence_frac.json -t FTS
python simulation_plot_trends.py -i $DIR/simulation_N${NUSERS}_trends.json -t trend
python simulation_plot_matrices.py -i $DIR/simulation_N${NUSERS}_mode101.json $DIR/simulation_N${NUSERS}_mode102.json

gzip $DIR/*.*

exit 0
###############################################
#for testing:
python simulation.py -c 1 -p mode=-1,N=200,numtrials=1 -o /tmp/simtest0 -d
python simulation.py -c 1 -p mode=-1,N=200,trend=0.05,numtrials=1 -o /tmp/simtest1 -d
python simulation.py -c 1 -p mode=-1,N=200,TF=0.05,numtrials=1 -o /tmp/simtest2 -d
python simulation.py -c 1 -p mode=-1,N=200,F=0.05,numtrials=1 -o /tmp/simtest3 -d
python simulation.py -c 1 -p mode=-101,N=200,numtrials=1 -o /tmp/simtest4 -d


