#!/usr/bin/env bash

# Check if the user provided the parent directory
if [ $# -lt 1 ]; then
    echo "Usage: $0 PARENT_DIR"
    exit 1
fi

parent_dir="$1"

# Loop through each subdirectory under parent_dir
for d in "$parent_dir"/*; do
    tmp_dir="$d/tmp"
    bag_file="$tmp_dir/recorded_topics.bag"
    
    # Check if tmp directory and bag file exist
    if [ -d "$tmp_dir" ] && [ -f "$bag_file" ]; then
        echo "Reindexing: $bag_file"
        rosbag reindex "$bag_file"
    fi
done