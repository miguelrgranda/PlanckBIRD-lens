#! /bin/bash

# Main bash script to launch the jobs.
# Three cases will be considered here.

# Case 1: no foregrounds case.
jid_no_fg=$(sbatch jobs/run_LiteBIRD_Planck_no_fg.sh)
echo $jid_no_fg
sbatch --dependency=afterok:${jid_no_fg##* } jobs/run_LiteBIRD_Planck_no_fg_serial.sh

# Case 2: s1d1f1a1co1 foregrounds case.
jid_s1_d1=$(sbatch --dependency=afterok:${jid_no_fg##* } jobs/run_LiteBIRD_Planck_s1_d1.sh)
echo $jid_s1_d1
sbatch --dependency=afterok:${jid_s1_d1##* } jobs/run_LiteBIRD_Planck_s1_d1_serial.sh

# Case 3: s5d10f1a1co3 foregrounds case.
jid_s5_d10=$(sbatch --dependency=afterok:${jid_no_fg##* } jobs/run_LiteBIRD_Planck_s5_d10.sh)
echo $jid_s5_d10
sbatch --dependency=afterok:${jid_s5_d10##* } jobs/run_LiteBIRD_Planck_s5_d10_serial.sh
