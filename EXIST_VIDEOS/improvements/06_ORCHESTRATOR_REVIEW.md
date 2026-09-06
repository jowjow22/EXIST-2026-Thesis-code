# Code review do orquestrador — implementação do modelo de vídeo melhorado

Revisão independente dos artefatos produzidos pelo Agente 4 (implementador),
contra as especificações dos relatórios 01–04. **Nada aqui foi aceito por
relato: cada item abaixo foi verificado rodando comando no repo.**

Data: 2026-08-07. Revisor: orquestrador do time.

---

## 1. Veredicto resumido

**O código está substancialmente melhor que o original e as correções
obrigatórias entraram de fato, não apenas na documentação.** 15 de 15 itens
verificáveis localmente passaram.

**Mas o objetivo "melhoria/otimização do modelo" NÃO está demonstrado — está
apenas plausível.** Nenhum treino rodou. Não há uma única métrica nova. O que
existe é: uma causa raiz confirmada (ARCH-01, verificada no output da célula
22 do notebook original) e uma correção que a endereça. Se a correção melhora
ICM, e quanto, é hipótese não testada. Ver seção 5.

---

## 2. Verificado e correto

| # | Item | Como verifiquei | Resultado |
|---|---|---|---|
| 1 | Originais intactos | `git diff --stat` nos 5 arquivos originais | sem alterações |
| 2 | Compila | `python3 -m py_compile video_core.py` | OK |
| 3 | Notebook válido | `json.load` + checagem nbformat | nbformat 4.5, 35 células |
| 4 | Sem outputs no notebook | varredura de `outputs`/`execution_count` | nenhuma célula suja |
| 5 | Sintaxe das células | `ast.parse` nas 17 células de código | nenhum erro |
| 6 | Sem execução no import | AST dos nós top-level de `video_core.py` | nenhuma chamada top-level |
| 7 | ARCH-01 implementado | `restore_videomae_qv_biases`, linha 1048 | presente, com falha alta e diagnóstico |
| 8 | `LABELS_2_1` morta | grep | removida; `TASK_A_PROB_ORDER` indexada de verdade |
| 9 | Selfcheck da Task A amarra as 2 pontas | leitura do corpo da função | sim — testa direção da loss E do formatter |
| 10 | Modality dropout por amostra | leitura do `forward` | `torch.rand(B)` — bug original (`torch.rand(1)`) corrigido |
| 11 | Faixas com taxa igual entre braços | álgebra + selfcheck Monte-Carlo | `clamp(floor(r/p), max=4)`, p exato por modalidade |
| 12 | Largura de fusão variável | `fusion_input_dim` + lista `parts` | 2560/2688 real, não zeros disfarçados |
| 13 | SWA removido | grep | só comentários explicando a remoção |
| 14 | Sem seleção por métrica reportada | grep | `load_best_model_at_end=False`, `metric_for_best_model` ausente |
| 15 | Seeds separados | grep | `SPLIT_SEED` fixo ≠ `INIT_SEED` |

Extras verificados: `MAX_LENGTH` 128→256 (ARCH-05); split congelado real e
consistente (`n_train=2019` + `n_val=505` = 2524, bate com o dataset);
exceção do `.gitignore` funciona (só `val_split_42.json` passa, não o
`Dataset/` inteiro).

---

## 3. Findings meus — não flagados por nenhum agente

### REV-01 (MÉDIO) — restrição do brief é insatisfazível; ninguém disse

O `CLAUDE.md` (linhas 106, 201-202) instrui: *"Ao mexer em LoRA + bf16,
confirmar chamada de `reset_lora_weights()` / `reset_new_layer_weights()`
pós-instanciação"*. Repassei isso aos 3 agentes no brief compartilhado.

Busca no repo inteiro:
```
CLAUDE.md:106  ... reset_lora_weights() e reset_new_layer_weights() devem ser chamados
CLAUDE.md:201  ... confirmar chamada de reset_lora_weights() /
CLAUDE.md:202  ... reset_new_layer_weights() pós-instanciação
```
**Essas funções não existem em lugar nenhum do código** — nem no
`Thesis_video_fixed.ipynb` original, nem no `memes_core.py`, nem no
`ablation/train_one.py`. Só na documentação.

`bf16=True` está ativo no notebook novo (célula 18), como estava no original.

**Isto NÃO é regressão** — o código novo preserva o comportamento do original,
que também usava `bf16=True` sem reset. Mas é uma restrição do projeto que
nenhum dos 3 agentes sinalizou como não-satisfazível, e é a **terceira**
afirmação obsoleta encontrada no `CLAUDE.md` nesta sessão (as outras duas:
`warmup_ratio` "removido" e dimensão de fusão 2592). Ação: decidir se a
mitigação nunca foi implementada ou foi removida, e corrigir o `CLAUDE.md`
de acordo.

### REV-02 (BAIXO) — regressão de custo não declarada no dropout por amostra

Original (`torch.rand(1)`, por batch):
```python
if drop_video:
    vid_feat = self._zeros(...)      # forward do VideoMAE PULADO
else:
    vid_out = self.videomae(...)
```
Novo (`torch.rand(B)`, por amostra):
```python
vid_feat = _mask(self.video_pool(self.videomae(pixel_values=frames)...), keep_video)
```
O forward do VideoMAE agora roda **sempre** e o resultado é mascarado depois.
Com dropout por amostra isso é inevitável (amostras diferentes são derrubadas
no mesmo batch), e a correção de granularidade vale o custo — mas o custo
existe: ~5% dos forwards de vídeo que antes eram pulados agora rodam. Dado que
o VideoMAE é o componente mais caro do forward, isso é da ordem de +5% no
tempo de treino, não zero. Não está declarado em nenhum relatório.

### REV-03 (INFORMATIVO) — `TEST_CASE` mudou de valor sem verificação possível

O implementador fixou `TEST_CASE = "EXIST2026"` (original: `"EXIST2025"`).
Para métricas de treino é irrelevante (predição e gold são gerados pelo mesmo
código, com a mesma string). Para submissão oficial é determinante: se o gold
do EXIST 2026 usar outra string, a avaliação volta vazia. O agente flagou
corretamente como pendente de verificação humana contra o gold oficial. Registro
aqui para não se perder: **é um item bloqueante de submissão, não de treino.**

---

## 4. Riscos que o implementador já fechou (crédito onde é devido)

Dois riscos que ele identificou e resolveu sem ser instruído:

1. **Split criado localmente, `Dataset/` não sincroniza com a VM** — sem
   guarda, a primeira run lá recriaria o split em silêncio, quebrando a
   premissa "as 9 runs partilham o mesmo val". Resolvido com `assert` na
   célula 4 e item pendente nº 0.
2. **Guarda `labels_a.ndim == 2` aplicada no lugar errado** — estava só na
   cópia local de `_metrics_task_a`; se disparasse, derrubaria `compute_metrics`
   inteiro, que é o oposto do que o ARCH-14 pede. Içada para `compute_metrics`.

Também descarregou o portão do §6.3 em vez de adiá-lo: rodou
`build_frozen_split` sobre as 2.524 linhas reais, estratificação por Task A
passou, categoria mais rara com 31 positivos.

---

## 5. O objetivo foi atingido? — avaliação honesta

**Sim, para "simplificar o código, encontrar bugs e inconsistências":**
- 1.418 linhas de notebook monolítico → módulo de 1.925 linhas com 16 seções
  + notebook de 35 células que só orquestra. (O módulo é maior em linhas
  porque absorveu `inference_video.py`/`ablation_harness.py` e porque
  comentários de decisão passaram a ser explícitos.)
- 4 bugs confirmados por evidência direta: bias do VideoMAE descartados
  (output da célula 22), 2 condições de ablação idênticas (md5 das linhas do
  CSV), seeds acopladas a split (linha 467), checkpoint lido de diretório
  errado (traceback salvo no `.ipynb`).

**Não, para "otimização/melhoria do modelo" no sentido de métrica:**
Nenhum treino rodou — não há GPU nesta sessão e o treino vive numa VM remota.
Zero métricas novas foram produzidas. A cadeia de raciocínio é:
ARCH-01 é fato verificado → a correção é plausível → **portanto o modelo deve
melhorar** é hipótese, não resultado. O tamanho do ganho é desconhecido e pode
ser pequeno.

**O que precisa acontecer para o objetivo ser atingido de fato:** rodar a
sequência de verificação pendente do `05_IMPLEMENTATION_NOTES.md` na VM,
começando pelo item 0 (guarda do split) e pela confirmação do caminho de
atributo do ARCH-01. Só depois de uma run de calibração de 15 épocas
pós-correção é que existe base para afirmar melhoria.

---

## 6. Ressalva de escopo que ultrapassa o código

Três achados independentes (SIMP-01, ARCH-20, OPT-06) atingem a tabela de
ablação do paper CEUR-WS **já submetido**: uma seed apresentada como agregado
de três, duas condições com flags idênticas, e braços de ablação partilhando
pesos treinados sempre com bio+viewer. Isso não é corrigível por código novo —
é decisão do autor e dos orientadores. Nenhuma ação foi tomada sobre o paper,
o `CLAUDE.md`, ou qualquer resultado publicado.
