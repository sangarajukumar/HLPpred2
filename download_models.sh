#!/usr/bin/env bash
# Download the HLPpred 2.0 models from Zenodo (https://doi.org/10.5281/zenodo.23474229),
# verify the checksums, join the split archives and extract them into the target directory.
# Usage: bash download_models.sh [target_dir]   (default: models)
set -euo pipefail

BASE_URL="https://zenodo.org/records/23474229/files"
DEST="${1:-models}"

mkdir -p "$DEST"
cd "$DEST"

cat > md5sums.txt <<'EOF'
8ac9df01f80d1d8f762339d714c3582c  HLPpred2_classification_DL.zip
cfa3fc3da0893b0a7e95e454aa882289  HLPpred2_classification_ML.zip.part00
35a64cf3a2867991d76a0050b81f6f54  HLPpred2_classification_ML.zip.part01
d215224afce795f3a6921c54e028d179  HLPpred2_classification_ML.zip.part02
694efea92a1714fe432bd857a7cad3de  HLPpred2_classification_woMERCI_DL.zip
c72ee541eb8a6ecc9d86860d29b6a5b5  HLPpred2_regression_DL.zip
ede41c29a97e0679be929e607e5ffe76  HLPpred2_regression_ML.zip.part00
e16066d7d10bf5ad826b788f96d21fb5  HLPpred2_regression_ML.zip.part01
41f9bf367bd5bccef2b7c93d4048b10d  HLPpred2_regression_ML.zip.part02
9dd875672537e384b9b4195efb311e79  HLPpred2_regression_ML.zip.part03
6c8ae351098036ed80e0ffdd2af025e1  HLPpred2_regression_ML.zip.part04
7e23867770605033665224d5af91b16a  HLPpred2_regression_ML.zip.part05
22de1dcfa8a9c5282ee86047b02bc914  HLPpred2_regression_ML.zip.part06
2e109516a5a3fa9a299b21ff177b8821  HLPpred2_regression_ML.zip.part07
aea11f4ce809d27919c7d4f5c08d0d6a  HLPpred2_regression_ML.zip.part08
fab175b605d7f708b97d1b1aa448bf36  HLPpred2_regression_ML.zip.part09
EOF

while read -r _ name; do
    [ -f "$name" ] || curl -fL --retry 3 -o "$name" "$BASE_URL/$name?download=1"
done < md5sums.txt

md5sum -c md5sums.txt

for task in classification regression; do
    cat HLPpred2_${task}_ML.zip.part* > HLPpred2_${task}_ML.zip
done
for archive in HLPpred2_*.zip; do
    unzip -q -o "$archive"
done

rm -rf __MACOSX
rm -f HLPpred2_*.zip HLPpred2_*.zip.part* md5sums.txt
echo "Models extracted to $(pwd)"
