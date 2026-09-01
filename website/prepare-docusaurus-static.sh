#!/usr/bin/env bash
set -euo pipefail

mkdir -p website/assets/images

copy_if_present() {
  local source=$1
  local destination=$2

  if [[ -f "$source" ]]; then
    cp -f "$source" "$destination"
  fi
}

copy_if_present \
  paper/sections/theoretical_background/imgs/fuked_manifold.png \
  website/assets/images/manifold_distortion.png
copy_if_present \
  paper/sections/results_discussion/imgs/fineweb_wandb.png \
  website/assets/images/training_curves.png
copy_if_present \
  paper/sections/results_discussion/imgs/mnist_prototypes.png \
  website/assets/images/mnist_prototypes.png
copy_if_present \
  paper/sections/results_discussion/imgs/linear.png \
  website/assets/images/linear_boundary.png
copy_if_present \
  paper/sections/results_discussion/imgs/yat.png \
  website/assets/images/yat_boundary.png
copy_if_present \
  paper/vector_viz_iclr/combined_heatmaps_2d.png \
  website/assets/images/combined_heatmaps_2d.png
copy_if_present \
  paper/vector_viz_iclr/combined_gradients.png \
  website/assets/images/combined_gradients.png

mkdir -p website/docusaurus/static/img/blog
cp -R website/assets/. website/docusaurus/static/img/blog/

mkdir -p website/docusaurus/static/paper
cp -f website/index.html website/nmn.html website/docusaurus/static/paper/
cp -R \
  website/css \
  website/js \
  website/assets \
  website/blog-pages \
  website/visualizations \
  website/docusaurus/static/paper/
