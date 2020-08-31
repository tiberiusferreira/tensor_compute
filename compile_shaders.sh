#!/bin/zsh
set -e # exit on errors
for file in $(find ./src -name '*.comp'); do
  file_dot_spv=${file%.*}.spv
  ./glslc -O "$file" -o "$file_dot_spv"
  echo "$file -> $file_dot_spv"
done

echo "SUCCESS!"
