#!/usr/bin/env bash
set -euo pipefail

: "${TENSORRT_RTX_URL:?TENSORRT_RTX_URL must point to an authorized TensorRT-RTX archive}"
: "${TENSORRT_RTX_SHA256:?TENSORRT_RTX_SHA256 must contain the archive SHA-256}"

workspace="${GITHUB_WORKSPACE:-$(pwd)}"
download_dir="$workspace/.tensorrt-download"
archive="$download_dir/tensorrt-rtx.archive"
extract_dir="$download_dir/extracted"
output_dir="$workspace/tensorrt-sdk"

rm -rf "$download_dir" "$output_dir"
mkdir -p "$download_dir" "$extract_dir" "$output_dir"

curl_args=(--fail --location --retry 5 --retry-all-errors --silent --show-error)
if [[ -n "${NVIDIA_API_KEY:-}" ]]; then
  curl_args+=(--header "Authorization: Bearer ${NVIDIA_API_KEY}")
fi
curl "${curl_args[@]}" "$TENSORRT_RTX_URL" --output "$archive"
echo "$TENSORRT_RTX_SHA256  $archive" | sha256sum --check --status

case "${TENSORRT_RTX_URL,,}" in
  *.zip|*.zip\?*)
    unzip -q "$archive" -d "$extract_dir"
    ;;
  *.tar.gz|*.tgz|*.tar.gz\?*) tar -xzf "$archive" -C "$extract_dir" ;;
  *.tar.zst|*.tzst|*.tar.zst\?*) tar --zstd -xf "$archive" -C "$extract_dir" ;;
  *)
    echo "Unsupported TensorRT-RTX archive URL; use .zip, .tar.gz, .tgz, or .tar.zst" >&2
    exit 2
    ;;
esac

dll="$(find "$extract_dir" -type f -iname 'tensorrt_rtx_*.dll' -print -quit)"
[[ -n "$dll" ]] || { echo "TensorRT runtime DLL was not found in the archive" >&2; exit 3; }
sdk_root="$(dirname "$(dirname "$dll")")"
for required in \
  "bin/tensorrt_rtx_1_6.dll" \
  "bin/tensorrt_onnxparser_rtx_1_6.dll" \
  "include/NvInfer.h" \
  "lib/tensorrt_rtx_1_6.lib" \
  "doc/Acknowledgements.txt" \
  "doc/README.txt"; do
  [[ -f "$sdk_root/$required" ]] || {
    echo "TensorRT-RTX archive is missing $required (detected root: $sdk_root)" >&2
    exit 4
  }
done
cp -a "$sdk_root/." "$output_dir/"
printf 'TensorRT-RTX SDK staged at %s\n' "$output_dir"
