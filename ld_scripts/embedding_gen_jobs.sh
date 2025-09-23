#!/bin/bash

# --- Configuration ---
# Set this flag to 'true' to use the custom separator and full dataset.
# Set to 'false' to use the default comma separator and subset data.
CUSTOM_SEPARATOR_FLAG=false

# Base directory for all outputs
OUTPUT_DIR_BASE="/opt/data/commonfilesharePHI/ldiao/ckd_project/ckd_embedding"
NUM_CPUS=4
TEMP_DIR="/tmp/data_chunks"
SCRIPT_PATH="./embedding_gen_jobs.py"

# Define file paths and delimiter based on the flag
if [ "$CUSTOM_SEPARATOR_FLAG" = true ]; then
    FULL_DATA_FILE="/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/CKD-Pull_v2.rpt"
    OUTPUT_DIR="${OUTPUT_DIR_BASE}_full_split"
    DELIMITER='$'
    echo "Using custom separator and full dataset."
else
    # The original configuration for subset data
    SUBSET_SIZE="10"
    FULL_DATA_FILE="/opt/data/commonfilesharePHI/slee/ckd-optum/patients_subset_${SUBSET_SIZE}.csv"
    OUTPUT_DIR="${OUTPUT_DIR_BASE}_subset_${SUBSET_SIZE}_split"
    DELIMITER=','
    echo "Using default separator and subset dataset."
fi

# Create necessary directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

# Get unique PatientIDs from the data file, skipping the header.
pids=$(cut -d"$DELIMITER" -f1 "$FULL_DATA_FILE" | tail -n +2 | sort -u)
pid_array=($pids)
total_pids=${#pid_array[@]}
chunk_size=$(( (total_pids + NUM_CPUS - 1) / NUM_CPUS ))

# Split data based on PatientID
echo "Splitting data into $NUM_CPUS chunks..."
for (( i=0; i<$NUM_CPUS; i++ )); do
    start_index=$(( i * chunk_size ))
    end_index=$(( start_index + chunk_size - 1 ))
    
    # Create an array of PIDs for this chunk
    chunk_pids=()
    for (( j=start_index; j<=end_index && j<total_pids; j++ )); do
        chunk_pids+=(${pid_array[j]})
    done

    # Create the chunk file for this job
    chunk_file="${TEMP_DIR}/chunk_${i}.csv"
    
    # Copy header from original file
    head -n 1 "$FULL_DATA_FILE" > "$chunk_file"

    # Append data for each PID in the chunk, using the correct delimiter
    for pid in "${chunk_pids[@]}"; do
        grep -E "^${pid}${DELIMITER}" "$FULL_DATA_FILE" >> "$chunk_file"
    done
    
    echo "Created chunk file: $chunk_file"
done

# Run jobs in parallel
echo "Starting $NUM_CPUS parallel jobs..."
for (( i=0; i<$NUM_CPUS; i++ )); do
    input_file="${TEMP_DIR}/chunk_${i}.csv"
    
    # Run the python script in the background, passing the custom separator flag if needed
    if [ "$CUSTOM_SEPARATOR_FLAG" = true ]; then
        python "$SCRIPT_PATH" --input_csv "$input_file" --output_dir "$OUTPUT_DIR" --cpu_id "$i" --custom_separator > "$OUTPUT_DIR/job_${i}.log" 2>&1 &
    else
        python "$SCRIPT_PATH" --input_csv "$input_file" --output_dir "$OUTPUT_DIR" --cpu_id "$i" > "$OUTPUT_DIR/job_${i}.log" 2>&1 &
    fi
done

# Wait for all background jobs to finish
wait

echo "All jobs finished. Check logs in $OUTPUT_DIR."

# Clean up temporary files
rm -rf "$TEMP_DIR"
echo "Temporary files cleaned up."