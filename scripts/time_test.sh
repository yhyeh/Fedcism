#!/bin/bash

#SECONDS=0

start=`date +%s`
sleep 1
end=`date +%s`

SECONDS=$( echo "$end - $start" | bc -l )
duration=$SECONDS
#H=$( echo "$duration / 3600" | bc -l )
#M=$( echo "$duration / 60" | bc -l )
#S=$( echo "$duration % 60" | bc -l )
#ELAPSED="Elapsed: $H hrs $M min $S sec"
echo process start time: ${start}
echo process end   time: ${end}
#echo elapse        time: ${ELAPSED}
echo duration: ${duration} sec, that is, 
echo "$(($duration / 3600)) hrs $(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
