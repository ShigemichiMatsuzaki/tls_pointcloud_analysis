PLOT_ID=$1
ITERATION=200

if [ -z "$PLOT_ID" ]; then
    PLOT_ID=""
fi

python3 open_space_analysis.py \
    --octomap-resolution 0.5 \
    --path-planning \
    --plot-id "${PLOT_ID}" \
    --iteration ${ITERATION} \
    --visualize \
    --region 1

    #--visualize \
python3 open_space_analysis.py \
    --octomap-resolution 0.5 \
    --path-planning \
    --plot-id "${PLOT_ID}" \
    --iteration ${ITERATION} \
    --region 0