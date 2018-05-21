#! /usr/bin/env bash
################################################################################
#
# oarsub -S ./word_tagging.sh
# https://github.com/ULHPC/launcher-scripts/blob/
# devel/bash/besteffort/launcher_besteffort.sh
#
################################################################################

#OAR -n word_tagging
#OAR -t besteffort
#OAR -t idempotent
#OAR -l walltime=3:00:00
#OAR -p gpumem>8000 and not gpumodel='p100'
#OAR -O /home/amensch/output/sdtw/word_tagging/%jobid%.log
#OAR -E /home/amensch/output/sdtw/word_tagging/%jobid%.log
#OAR --checkpoint 60
#OAR --signal 12

if type oarprint &> /dev/null; then
    GPUID=$(oarprint host -P gpuid)
else
    GPUID=0
fi

unset CUDA_VISIBLE_DEVICES
#export CUDA_VISIBLE_DEVICES=${GPUID}


# Unix signal sent by OAR, SIGUSR1 / 10
CHKPNT_SIGNAL=12

# exit value for job resubmission
EXIT_UNFINISHED=99

#####################################
#                                   #
#   Environment                     #
#                                   #
#####################################
if [ -f  $HOME/.bashrc ]; then
    .  $HOME/.bashrc
fi

OUTPUT=$HOME/output/sdtw/word_tagging
mkdir -p ${OUTPUT}

##########################################
# Run the job
WORK=$HOME/work/repos/soft-dtw-pp/exps
cd ${WORK}
#bash ${WORK}/mongo_pipe.sh || True

python -u ${WORK}/word_tagging.py $@ &
PID=$!

trap "kill -$CHKPNT_SIGNAL $PID; wait $PID; exit $EXIT_UNFINISHED" $CHKPNT_SIGNAL

wait $PID
RET=$?

exit $RET

