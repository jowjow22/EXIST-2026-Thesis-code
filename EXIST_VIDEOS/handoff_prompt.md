Fale sempre em português do Brasil (preferência registrada do autor).

Antes de qualquer coisa, leia por completo:
`/Users/jonathansantosdasilva/Documents/projects/VM-THESIS/EXIST_VIDEOS/improvements/context-report-clip-branch-and-head-probing.md`

Esse arquivo é o handoff completo de uma sessão anterior sobre o braço CLIP (item 10 do
doc 17) do sistema de vídeo do EXIST 2026 Task 3. Cobre: as 4 runs de treino do braço,
duas hipóteses de correção testadas e refutadas (overtraining/argmin-de-época e
desequilíbrio de capacidade do encoder 4→2), a decisão final de reportar o braço no
estado FINAL da época 15 (não argmin), o debate época 2 vs 7 vs 15 (resolvido, mantida a
15), e o resultado do probing de heads que acabou de terminar — INCONCLUSIVO por falha
de controle, rota fechada por medição, não consertada.

Também relevantes, se precisar de mais detalhe além do que está no context report:
- `EXIST_VIDEOS/improvements/16_VIDEO_INVESTIGATION_RESULTS.md` — investigação original
- `EXIST_VIDEOS/improvements/17_RETRAIN_IMPROVEMENT_OPTIONS.md` — item 10 = CLIP
- `EXIST_VIDEOS/improvements/18_CLIP_VIDEO_BRANCH_SPEC.md` — §9 tem o histórico completo
  e verificado das 4 runs, com a tabela de 15 épocas e as métricas de cada checkpoint
- `EXIST_VIDEOS/improvements/20_EPOCH_SELECTION_TASK_C.md` — pesquisa de literatura MTL
  sobre a divergência eval_loss/Task C, opções 1-6 (a opção 5 é a única que resta)
- `EXIST_VIDEOS/improvements/21_HEAD_PROBING_PREREGISTRATION.md` — pré-registro do
  probing + seção 6 "Resultado (2026-08-15)" com o veredito completo
- `EXIST_VIDEOS/video_core_clip.py` — overlay do encoder CLIP (`UNFROZEN_VISION_LAYERS=4`,
  revertido, não mexer de novo)
- `EXIST_VIDEOS/Thesis_video_clip.ipynb` — notebook de treino do braço
- `EXIST_VIDEOS/probe_head_training.ipynb` — notebook do probing (já rodado, resultado
  documentado)

**Estado resumido:** braço CLIP encerrado no estado final da época 15 (4 camadas). Duas
hipóteses de correção pro colapso do ramo de texto foram testadas e refutadas. O probing
de heads (tentativa mais recente e mais barata de decidir se a Task C está sub-treinada
ou se a representação do trunk não carrega sinal suficiente dela) deu INCONCLUSIVO porque
os controles A/B falharam — protocolo não confiável, rota fechada, não consertada.

**Pergunta pendente pro autor, ainda sem resposta:** aplicar a opção 5 do doc 20
(retreino completo do trunk CLIP com `LOSS_W_C` repesado de 0,7 para 4,2 — valor
principiado, ~3,5h de GPU, é realocação: subir C deve custar A e/ou B) ou aceitar o
estado atual do braço CLIP como resultado final, sem mais intervenções.

**Abra esta sessão perguntando essa decisão ao autor antes de escrever ou rodar qualquer
código.** Não presuma qual das duas opções ele prefere.
