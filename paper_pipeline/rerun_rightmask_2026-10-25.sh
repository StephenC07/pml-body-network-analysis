#!/bin/bash
# Rerun of the Fig. 2 network pipeline after masking the 60 px right-edge artefact strip in all five tomograms (25 Oct 2026).
cd /Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/pml-network-density-accounted
set -e
for step in "fibre_after_blob_removal.py isonet" "fibre_after_blob_removal.py raw" "controls_fibre_denominator_bandpass.py isonet" "controls_fibre_denominator_bandpass.py raw" "boundary_profile.py" "boundary_profile_rm15.py" "boundary_shift_sweep.py" "fig2bc_panels.py" "fig2_assemble_with_render.py" "fig_supp2_assemble_600dpi.py"; do
  echo "=== $(date '+%H:%M:%S') START $step"; python3 $step > "results_v2/rerun_rightmask_$(echo $step | tr ' .' '__').log" 2>&1; echo "=== $(date '+%H:%M:%S') DONE $step"
done
echo ALL DONE
