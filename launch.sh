  
#!/bin/bash

rm -f yml/tmp.yml
cp "$YAML" yml/tmp.yml

for i in $(seq "$NS" "$NE"); do
  echo "Submitting job with workers=$i"   
  rm -f yml/run_$i.yml

  # Update YAML
  sed "/^[[:space:]]\{1,\}workers:[[:space:]]*$/ {n; s/^[[:space:]]*- .*/      - $i/;}" yml/tmp.yml > yml/run_$i.yml

  if [ -z "$prev_jobid" ]; then
    # First job (no dependency)
    jobid=$(sbatch -x k003-009 --nodes="$i" "$JOB_SCRIPT" | awk '{print $4}')
  else
    # Subsequent jobs depend on previous one
    # jobid=$(sbatch -x k003-009 --nodes="$i" --dependency=afterok:$prev_jobid "$JOB_SCRIPT" | awk '{print $4}')
    jobid=$(sbatch -x k003-009 --nodes="$i" "$JOB_SCRIPT" | awk '{print $4}')
  fi

  echo "Submitted job $jobid (depends on $prev_jobid)"
  prev_jobid=$jobid
done

rm -f yml/tmp.yml
