#!/usr/bin/env bash
# Generate a synthetic EHR population with Synthea (needs Java 17+).
# Usage: bash scripts/01_generate_synthea.sh [N_PATIENTS] [SEED]
set -euo pipefail
N=${1:-1000}
SEED=${2:-42}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="$ROOT/data/tools"; OUT="$ROOT/data/synthea"
mkdir -p "$TOOLS" "$OUT"
JAR="$TOOLS/synthea-with-dependencies.jar"
if [ ! -f "$JAR" ]; then
  echo "Downloading Synthea..."
  curl -L -o "$JAR" https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar
fi
cd "$TOOLS"
java -Xmx4g -jar "$JAR" -s "$SEED" -cs "$SEED" -p "$N" -a 35-80 \
  --exporter.csv.export=true --exporter.fhir.export=false \
  --exporter.hospital.fhir.export=false --exporter.practitioner.fhir.export=false \
  --exporter.baseDirectory="$OUT" Massachusetts
echo "CSV files written to $OUT/csv"
