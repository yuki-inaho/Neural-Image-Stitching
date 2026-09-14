#!/usr/bin/env bash
# Download the official NIS pretrained weights from Google Drive and extract
# them into <repo>/pretrained/:
#   1. NIS_enhancing.pth : Enhanced Stitching (Stage 1)
#   2. NIS_blending.pth  : Enhanced & Blended Stitching (Stage 1 & 2)
#   3. ihn.pth           : reproduced Homography Estimator
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FILE_ID="1sdfquwxhKLq2aBGGdtiu8_SM-g-aDUtM"
ARCHIVE="$REPO_ROOT/.cache/NIS_pretrained.zip"
ZIP_SHA256="33b4b72b94cca90957c3ca8ef051bc6b5ead64eb84bdb5fda5507bbb20ce39c1"

declare -A PTH_SHA256=(
    [NIS_blending.pth]="53d15939589b4af41735e71e3d26fce432f18a9cbe1c80490322d56d9b5d48ea"
    [NIS_enhancing.pth]="48682884be3e2264bd2be8031df025ea32d280f36e9e07e627cecf243055818c"
    [ihn.pth]="e53a3158a9ceecd561617160a1983e05a61116690a4763c3419f2f45da5952da"
)

verify_pth() {
    local name="$1" file="$REPO_ROOT/pretrained/$1" want="${PTH_SHA256[$1]}"
    [ -f "$file" ] || return 1
    local got
    got="$(sha256sum "$file" | cut -d' ' -f1)"
    [ "$got" = "$want" ]
}

all_present=1
for name in "${!PTH_SHA256[@]}"; do
    verify_pth "$name" || all_present=0
done
if [ "$all_present" -eq 1 ]; then
    echo "==> pretrained weights already present and verified in $REPO_ROOT/pretrained"
    exit 0
fi

mkdir -p "$REPO_ROOT/.cache"
if [ -f "$ARCHIVE" ] && [ "$(sha256sum "$ARCHIVE" | cut -d' ' -f1)" != "$ZIP_SHA256" ]; then
    echo "==> cached archive checksum mismatch, re-downloading"
    rm -f "$ARCHIVE"
fi
if [ ! -f "$ARCHIVE" ]; then
    echo "==> downloading NIS_pretrained.zip (~63 MiB) from Google Drive"
    gdown "https://drive.google.com/uc?id=$FILE_ID" -O "$ARCHIVE"
fi

echo "==> verifying archive checksum"
echo "$ZIP_SHA256  $ARCHIVE" | sha256sum -c -

echo "==> extracting into $REPO_ROOT"
python -c "import zipfile; zipfile.ZipFile('$ARCHIVE').extractall('$REPO_ROOT')"

for name in "${!PTH_SHA256[@]}"; do
    verify_pth "$name" || { echo "error: checksum mismatch for pretrained/$name" >&2; exit 1; }
    echo "    ok pretrained/$name"
done
echo "==> pretrained weights ready"
