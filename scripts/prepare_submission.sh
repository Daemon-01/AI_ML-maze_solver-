#!/bin/bash
# Package final deliverables

echo "Preparing submission..."

# Create deliverables directory
mkdir -p deliverables/supplementary_materials

# Archive figures
zip -r deliverables/supplementary_materials/all_figures.zip figures/

# Archive videos
zip -r deliverables/supplementary_materials/demo_videos.zip videos/

# Archive experimental data
zip -r deliverables/supplementary_materials/experimental_data.zip results/

echo "Submission package ready!"
