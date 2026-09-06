#!/usr/bin/env bash
# ============================================================================
# preserve_checkpoint.sh — baixa da VM os artefatos NÃO REGENERÁVEIS da Task 3
# e grava um checkpoint datado e verificável na máquina local.
#
# RODAR NO MAC (local), não na VM. Puxa por SSH, não pelo mount SSHFS.
#
#   ./preserve_checkpoint.sh              # tiers 1+2+3 (padrão, ~415 MB)
#   ./preserve_checkpoint.sh 1            # só o tier 1 (texto/código, ~5 MB)
#   ./preserve_checkpoint.sh 2            # tiers 1+2 (+ pesos e caches)
#   ./preserve_checkpoint.sh verify       # reverifica o último checkpoint por sha256
#
# Por que existe: o repositório "local" é um mount SSHFS da MESMA árvore da VM.
# Não é backup — é a mesma cópia vista de outro lugar. Um `rm` na VM apaga as
# duas. Os frames v1 já foram perdidos assim (problema V6).
#
# Compatível com bash 3.2 (o que vem no macOS): sem namerefs, sem `mapfile`.
# ============================================================================
set -euo pipefail

VM_HOST="a67606@172.31.16.19"
VM_ROOT="/home/a67606/Thesis/EXIST_VIDEOS"
DEST_ROOT="${HOME}/thesis-checkpoints"
STAMP="$(date +%Y-%m-%d_%H%M)"
DEST="${DEST_ROOT}/${STAMP}"
TIER="${1:-all}"

# ---------------------------------------------------------------------------
# TIER 1 — texto, código, métricas. Insubstituível, minúsculo. Sempre baixado.
# ---------------------------------------------------------------------------
TIER1="
video_core.py
improvements/
analysis/probe_report.txt
analysis/audio_durations.csv
analysis/truncation_scan_v2.csv
analysis/frames_zero_scan.json
analysis/probe_cache_B15/task_b_flips.csv
Dataset/val_split_42.json
probe_video_contribution.ipynb
Thesis_video_baseline.ipynb
Thesis_video_improved.ipynb
extract_video_v2.ipynb
extract_audio_v2.ipynb
verify_extraction_v2.ipynb
test.ipynb
verify_seeding.py
runs_baseline/
predictions/
"

# ---------------------------------------------------------------------------
# TIER 2 — pesos treinados e caches de sonda. Re-treináveis, mas caros.
# `tokenizer.json` fica de fora de propósito: 17 MB × 5 runs, idênticos entre si
# e baixáveis de `xlm-roberta-large`. Não é artefato de experimento.
# ---------------------------------------------------------------------------
TIER2="
run_B2_bio_viewer_seed42_A/
run_B2_bio_viewer_seed42_A_prime/
run_B2_bio_viewer_seed42_B/
run_B2_bio_viewer_seed42_B15/
run_B2_bio_viewer_seed42_CAL/
analysis/probe_cache_B15/
"

# ---------------------------------------------------------------------------
# TIER 3 — referências de extração. NÃO REGENERÁVEIS: os frames v1 completos
# foram apagados; `frames_v1_sample/` (20+5) é tudo o que resta deles, e é o
# insumo do teste de drift T4a. `audio_features/{train,test}` é a v1 preservada.
# ---------------------------------------------------------------------------
TIER3="
frames_v1_sample/
audio_features/train/
audio_features/test/
"

pull () {
  label="$1"; shift
  paths="$*"
  echo ""
  echo "── ${label} ──────────────────────────────────"
  for p in $paths; do
    # Barra final NÃO pode sobreviver: `rsync src/ dest/` copia o CONTEÚDO de src,
    # e `improvements/` acabaria despejado solto dentro de EXIST_VIDEOS/.
    # Sem a barra, o rsync recria o próprio diretório dentro do destino.
    p="${p%/}"
    target="${DEST}/EXIST_VIDEOS/$(dirname "$p")"
    mkdir -p "$target"
    printf '  %-52s' "$p"
    if rsync -az --partial \
             --exclude 'tokenizer.json' \
             --exclude '._*' --exclude '.DS_Store' \
             --exclude '.ipynb_checkpoints' \
             "${VM_HOST}:${VM_ROOT}/${p}" "${target}/" 2>/dev/null; then
      echo "ok"
    else
      echo "⚠ FALHOU / inexistente na VM"
      FAILED="${FAILED}${p} "
    fi
  done
}

if [ "$TIER" = "verify" ]; then
  if [ ! -d "${DEST_ROOT}/latest" ]; then
    echo "❌ nenhum checkpoint em ${DEST_ROOT}/latest — rodar sem argumento primeiro."
    exit 1
  fi
  echo "verificando ${DEST_ROOT}/latest por CONTEÚDO (sha256)."
  echo "o mtime não serve: este projeto já registrou mtime mentiroso no mount."
  cd "${DEST_ROOT}/latest"
  if shasum -a 256 -c MANIFEST.sha256 > /tmp/_ck_verify.txt 2>&1; then
    echo "✅ todos os $(wc -l < MANIFEST.sha256 | tr -d ' ') arquivos conferem"
  else
    echo "❌ divergências:"
    grep -v ': OK$' /tmp/_ck_verify.txt || true
    exit 1
  fi
  exit 0
fi

FAILED=""
echo "checkpoint  → ${DEST}"
echo "origem      → ${VM_HOST}:${VM_ROOT}"
mkdir -p "$DEST"

pull "TIER 1 — texto, código, métricas (~5 MB)" $TIER1

if [ "$TIER" = "all" ] || [ "$TIER" = "2" ] || [ "$TIER" = "3" ]; then
  pull "TIER 2 — pesos e caches (~160 MB)" $TIER2
fi

if [ "$TIER" = "all" ] || [ "$TIER" = "3" ]; then
  pull "TIER 3 — referências v1, NÃO regeneráveis (~250 MB)" $TIER3
fi

echo ""
echo "── manifest sha256 ───────────────────────────"
cd "$DEST"
find . -type f ! -name 'MANIFEST.sha256' -print0 | sort -z \
  | xargs -0 shasum -a 256 > MANIFEST.sha256
echo "  $(wc -l < MANIFEST.sha256 | tr -d ' ') arquivos · $(du -sh "$DEST" | cut -f1)"

ln -sfn "$DEST" "${DEST_ROOT}/latest"

echo ""
if [ -n "$FAILED" ]; then
  echo "⚠ NÃO baixados: ${FAILED}"
  echo "  checkpoint está INCOMPLETO — resolver antes de apagar qualquer coisa na VM."
  exit 1
fi
echo "✅ checkpoint completo em ${DEST}"
echo "   verificar depois com:  $0 verify"
