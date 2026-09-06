#!/usr/bin/env bash
# Junta os ficheiros de contexto da tese numa pasta única, fora do repo/mount, para
# arrastar de uma vez para o projeto do Claude Desktop.
#
#   ./collect_context_files.sh              -> núcleo + experimentos fechados (14)
#   ./collect_context_files.sh --all        -> os 14 + auditoria/extração (19)
#   ./collect_context_files.sh --all ~/tmp/ctx   -> destino alternativo
#
# Ficheiros `._*` (resíduos AppleDouble do macOS) nunca são copiados.
# Ver PROJECT_CONTEXT_PROMPT.md para o que cada ficheiro é e por que está na lista.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMP="$REPO/EXIST_VIDEOS/improvements"
THESIS="$REPO/EXIST_VIDEOS/thesis"
ABL="$REPO/EXIST_MEMES/ablation"

INCLUDE_EXTRA=0
DEST="$HOME/Desktop/thesis_context"
for arg in "$@"; do
  case "$arg" in
    --all) INCLUDE_EXTRA=1 ;;
    -*)    echo "opção desconhecida: $arg" >&2; exit 2 ;;
    *)     DEST="$arg" ;;
  esac
done

# --- Núcleo (9) ---
FILES=(
  "$REPO/CLAUDE.md"
  "$IMP/24_FINAL_BASELINE_DECISION.md"
  "$IMP/09_BASELINE_RESULTS.md"
  "$IMP/16_VIDEO_INVESTIGATION_RESULTS.md"
  "$IMP/18_CLIP_VIDEO_BRANCH_SPEC.md"
  "$IMP/13_NOTAS_PARA_TESE.md"
  "$THESIS/ch_video_task3_draft.md"
  "$ABL/ABLATION_REPORT.md"
  "$ABL/ABLATION_OVERVIEW.md"
)

# --- Experimentos fechados (5) ---
FILES+=(
  "$IMP/17_RETRAIN_IMPROVEMENT_OPTIONS.md"
  "$IMP/20_EPOCH_SELECTION_TASK_C.md"
  "$IMP/21_HEAD_PROBING_PREREGISTRATION.md"
  "$IMP/22_HEAD_FREEZE_PREREGISTRATION.md"
  "$IMP/23_CALIBRATION_AND_NAN_RESULTS.md"
)

# --- Auditoria e extração (5), só com --all ---
if [ "$INCLUDE_EXTRA" -eq 1 ]; then
  FILES+=(
    "$IMP/04_RECONCILIATION.md"
    "$IMP/05_IMPLEMENTATION_NOTES.md"
    "$IMP/07_ARCH04_FULL_POPULATION.md"
    "$IMP/14_VIDEO_CONTRIBUTION_ANALYSIS.md"
    "$IMP/10_EXTRACTION_FIXES_SPEC.md"
  )
fi

mkdir -p "$DEST"

copied=0
missing=()
for f in "${FILES[@]}"; do
  base="$(basename "$f")"
  case "$base" in ._*) continue ;; esac      # nunca copiar AppleDouble
  if [ -f "$f" ] && [ -s "$f" ]; then
    cp "$f" "$DEST/$base"
    copied=$((copied + 1))
  else
    missing+=("$base")
  fi
done

# O próprio prompt de instruções vai junto, para não ficar a depender do mount.
[ -f "$THESIS/PROJECT_CONTEXT_PROMPT.md" ] && cp "$THESIS/PROJECT_CONTEXT_PROMPT.md" "$DEST/"

echo "destino : $DEST"
echo "copiados: $copied ficheiros ($( [ "$INCLUDE_EXTRA" -eq 1 ] && echo 'núcleo + fechados + auditoria' || echo 'núcleo + fechados' ))"
echo "tamanho : $(du -sh "$DEST" | cut -f1)  ·  $(cat "$DEST"/*.md | wc -l | tr -d ' ') linhas"

if [ "${#missing[@]}" -gt 0 ]; then
  echo
  echo "⚠ NÃO encontrados (ou vazios) — verificar antes de subir:"
  printf '   %s\n' "${missing[@]}"
fi

echo
echo "Próximo: abrir $DEST, selecionar tudo e arrastar para o projeto no Claude Desktop."
echo "As instruções do projeto estão na secção 2 de PROJECT_CONTEXT_PROMPT.md."
