# 24 — Baseline final do sistema de vídeo (Task 3)

**Decisão do autor, 2026-08-16.** Encerra o ciclo de experimentação do sistema de vídeo
iniciado na auditoria de 2026-08-07. Nenhum treino adicional está previsto.

---

## 1. A decisão

**Reportar DOIS sistemas**, não um:

| papel | run | justificação |
|---|---|---|
| **Sistema principal** | `run_B2_bio_viewer_seed42_CLIP` | melhores métricas em 5 das 6 |
| **Referência de robustez** | `run_B2_bio_viewer_seed42_B15` | melhor C soft; apoia-se no ramo com sinal causal |

A divergência entre os dois **é** um achado da tese, não um detalhe de reporte — ver §3.

---

## 2. Os números (estado final, mesmo split congelado, `SEED=42`)

| métrica | `CLIP` | `B15` | Δ (CLIP − B15) |
|---|---|---|---|
| A ICMNorm hard | **0,5741** | 0,5536 | **+0,0205** |
| A ICMSoftNorm soft | **0,5172** | 0,5058 | **+0,0114** |
| B ICMNorm hard | **0,4986** | 0,4968 | +0,0018 |
| B ICMSoftNorm soft | **0,4122** | 0,3947 | **+0,0175** |
| C ICMNorm hard | **0,3640** | 0,3436 | **+0,0204** |
| C ICMNorm hard, limiares ajustados | **0,3697** | 0,3171 | **+0,0526** |
| C ICMSoftNorm soft | 0,1368 | **0,1514** | **−0,0146** |

⚠ A comparação é controlada em tudo menos o encoder (mesmo split, seed, protocolo de
época, áudio v2, frames v2), mas **não é uma ablação pareada** — há uma divergência de
interpolação declarada e não resolvida (frames extraídos com resize bilinear do VideoMAE,
CLIP usa bicubic por padrão; ver doc 18). Ler como comparação de teto, não como ablação.

⚠ As métricas de Task C hard com limiares ajustados vêm de ~metade do val (n=253),
enquanto as demais vêm do val inteiro. Não colocar lado a lado sem essa nota.

---

## 3. Por que dois sistemas, e não só o melhor

O CLIP ganha em métrica **apoiando-se no canal errado**. A sonda causal por permutação
mede a dependência de cada slot da fusão:

| | `‖Δ_texto‖` | leitura |
|---|---|---|
| `B15` | ~18,6 | o texto domina a decisão |
| `CLIP` | **0,785** | ramo de texto praticamente colapsado (~24× menor) |

E a mesma investigação (doc 16) mostrou que o slot de vídeo carrega **formato/registo da
fonte** — correlacionado com a classe de B — e não conteúdo. Um sistema que troca o canal
de conteúdo (transcrição) por um canal de formato pode generalizar **pior** no teste real
mesmo com validação melhor.

Isto não é especulação retrospetiva: está registado em doc 18 §9.4 como argumento de
robustez contra o braço, escrito antes desta decisão.

**Consequência para a tese:** o par (CLIP, B15) é a evidência de que *métrica de validação
agregada e dependência causal podem apontar em direções opostas* — um contributo
analítico, na mesma família do contributo negativo já planeado sobre normalização
per-viewer.

---

## 4. O que NÃO é baseline, e porquê

| run | estatuto |
|---|---|
| `CAL` | idêntica ao B15 (mesmos números) |
| `A`, `A_prime`, `B` | braços de atribuição da re-extração (isolam ARCH-01 e áudio); C soft ~0,06 |
| `CLIP_L2` | hipótese de capacidade do encoder — **refutada** (doc 18 §9.2) |
| `CLIP_WC42` | repesagem `LOSS_W_C` — **refutada** (doc 18 §9.5) |
| `CLIP_FT5_FREEZEAB` / `CLIP_FT5_CTRL` | freeze de heads A/B + controlo — **refutada** (doc 22 §6) |

Um braço com resultado negativo entra na tese como experimento, **nunca** como sistema
reportado.

---

## 5. Eixos declarados fechados

Nenhum destes reabre sem evidência nova (não sem uma ideia nova):

- Seleção de época que não seja "estado final" ou argmin pré-registado (doc 18 §9.2)
- `UNFROZEN_VISION_LAYERS` ≠ 4 (doc 18 §9.2)
- Realocação de treino entre tasks — repesagem de loss ou freeze de heads (doc 22 §6.5)
- Probing de heads sobre trunk congelado (doc 21 §6.3)
- Calibração pós-treino por NLL (doc 23 §2) — a variante ICM-driven fica registada como
  não executada, com teto estimado de +0,03–0,06 e **só** em A soft (doc 23 §2.3)

---

## 6. Pendências que sobrevivem a esta decisão

Nenhuma bloqueia o trabalho seguinte:

1. Sonda causal sobre os artefatos WC42 e fase 2 — nunca executada; o rascunho da tese
   não faz nenhuma afirmação causal sobre esses braços (marcador em aberto).
2. `probe_whisper_attention_padding.ipynb` (item 11, doc 19) — pronto, nunca executado.
   Diagnóstico, não correção.
3. Os 6 marcadores `[NÚMERO A CONFIRMAR]` de `thesis/ch_video_task3_draft.md`.

---

**Próxima fase do projeto:** sistema de confiança de anotadores. O sistema de vídeo entra
nessa fase como componente congelado.
