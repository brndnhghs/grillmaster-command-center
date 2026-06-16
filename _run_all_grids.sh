#!/bin/bash
cd /Users/admin/Documents/GitHub/grillmaster-command-center

echo "=== Starting 22 grid generations ==="
echo ""

run_grid() {
  local method=$1
  local param=$2
  local json_file="_grid_params/${method}_${param}.json"
  echo "[$(date +%H:%M:%S)] Running grid for method $method ($param)..."
  python3 _param_grid.py "$method" "$param" "$(cat "$json_file")" 2>&1
  echo ""
}

# Methods 22-23, 29
run_grid 22 font_size
run_grid 23 spread
run_grid 29 points

# Methods 37-48
run_grid 37 levels
run_grid 38 max_offset
run_grid 39 colors
run_grid 40 threshold
run_grid 41 radius
run_grid 42 gamma
run_grid 43 points
run_grid 44 circle_count
run_grid 47 shape_count
run_grid 48 ring1_center

# Methods 57, 59, 63-65
run_grid 57 amplitude
run_grid 59 corruption
run_grid 63 thread_step
run_grid 64 dot_size
run_grid 65 freq1

# Methods 74, 76-77, 80
run_grid 74 strength
run_grid 76 bits
run_grid 77 sigma
run_grid 80 tile_size

echo "=== All done at $(date) ==="
echo ""
echo "=== Generated grid files ==="
ls -la grid_*.png 2>/dev/null | awk '{print $NF}'