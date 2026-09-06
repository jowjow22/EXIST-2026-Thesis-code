"""ablation/config.py

Define a configuração de cada run de ablação (`AblationConfig`), a baseline
(`get_baseline_config`), e a suite completa de ablações (`ABLATION_SUITE`).

Cada entrada de `ABLATION_SUITE` é um template de config (seed = baseline).
`ablation/runner.py` expande cada template para as 3 seeds [42, 7, 2025] via
`dataclasses.replace(config, seed=..., name=...)` antes de rodar.

NÃO usa warmup_ratio em lado nenhum -- só warmup_steps (float = fração dos
passos totais, inteiro = nº de passos; ver CLAUDE.md, transformers==5.6.0 na VM).
"""

from dataclasses import dataclass, replace


SEEDS = [42, 7, 2025]

# Seed da permutação do demo_vec (ABL-INJ-CTRL). FIXA e independente de
# `config.seed` de propósito: as 3 seeds de treino têm de ver EXATAMENTE a mesma
# permutação, senão a comparação pareada seed-a-seed contra `inj_headinject_base`
# mistura duas fontes de variação e deixa de medir o que interessa.
DEMO_SHUFFLE_SEED = 1234


@dataclass(frozen=True)
class AblationConfig:
    name: str

    seed: int = 42
    num_epochs: int = 30
    warmup_steps: float = 0.1          # float [0,1) = fração; int = nº de passos. NUNCA warmup_ratio.
    task_a_weight: float = 2.0         # ABL-3
    unfrozen_vision_layers: int = 4    # ABL-4 (0 = CLIP totalmente congelado)
    use_vision: bool = True            # ABL-5
    use_demographics: bool = True      # ABL-5
    demo_feature_set: str = "base"     # ABL-2: "base" (66-d) ou "extended" (+gender+age)
    demo_norm_mode: str = "sum"        # ABL-6: "sum" ou "l2"
    demo_injection_mode: str = "current"      # ABL-INJ: "current" | "renorm" | "head_inject"

    # ABL-INJ: roda `memes_core.diagnose_task_c_collapse` na avaliação final e grava
    # o JSON em `ablation/diagnostics/`. É um campo de CONFIG (e não uma flag do
    # runner) de propósito: ele entra em `config_is_baseline_equivalent`, portanto
    # `inj_current_base` -- que fora isso é idêntico à baseline -- deixa de ser
    # "reaproveitável" e TREINA de verdade. Sem isso o runner reusaria a linha da
    # FASE 0, nenhum modelo existiria em memória e o diagnóstico nunca correria
    # justamente na config-âncora.
    run_collapse_diagnostics: bool = False

    # ABL-INJ-CTRL: embaralha o demo_vec entre itens (derangement de permutação
    # fixa) preservando a arquitetura. Isola "o ganho do head_inject vem da
    # demografia" de "vem das 512 dimensões extras nas heads".
    demo_shuffle: bool = False


def get_baseline_config(name="baseline"):
    """Defaults atuais do código: num_epochs=30, warmup_steps=0.1 (10%, já correto
    em transformers==5.6.0 -- não é um bug, ver CLAUDE.md)."""
    return AblationConfig(name=name)


BASELINE = get_baseline_config()


def _variant(name, **overrides):
    return replace(BASELINE, name=name, **overrides)


# ============================================================================
# ABL-1 -- schedule de warmup
# ============================================================================
ABL_1_WARMUP = [
    _variant("abl1_config_A_warmup_0.1", warmup_steps=0.1),   # = baseline
    _variant("abl1_config_B_warmup_0.0", warmup_steps=0.0),
    _variant("abl1_config_C_warmup_0.2", warmup_steps=0.2),
]

# ============================================================================
# ABL-2 -- demo_vec expandido (gender + age)
# ============================================================================
ABL_2_DEMO_VEC = [
    _variant("abl2_config_A_demo_base",     demo_feature_set="base"),      # = baseline
    _variant("abl2_config_B_demo_extended", demo_feature_set="extended"),
]

# ============================================================================
# ABL-3 -- peso da loss da Task A
# ============================================================================
ABL_3_TASK_A_WEIGHT = [
    _variant("abl3_config_A_taskaw_2.0", task_a_weight=2.0),  # = baseline
    _variant("abl3_config_B_taskaw_1.0", task_a_weight=1.0),
    _variant("abl3_config_C_taskaw_3.0", task_a_weight=3.0),
]

# ============================================================================
# ABL-4 -- camadas do CLIP descongeladas
# ============================================================================
ABL_4_UNFROZEN_VISION = [
    _variant("abl4_config_A_unfrozen_4", unfrozen_vision_layers=4),  # = baseline
    _variant("abl4_config_B_unfrozen_0", unfrozen_vision_layers=0),
    _variant("abl4_config_C_unfrozen_8", unfrozen_vision_layers=8),
]

# ============================================================================
# ABL-5 -- modalidades (visão / demografia)
# ============================================================================
ABL_5_MODALITIES = [
    _variant("abl5_config_A_vision_demo",    use_vision=True,  use_demographics=True),   # = baseline
    _variant("abl5_config_B_vision_only",    use_vision=True,  use_demographics=False),
    _variant("abl5_config_C_demo_only",      use_vision=False, use_demographics=True),
    _variant("abl5_config_D_text_only",      use_vision=False, use_demographics=False),
]

# ============================================================================
# ABL-6 -- normalização do demo_vec (soma vs L2)
# ============================================================================
ABL_6_DEMO_NORM = [
    _variant("abl6_config_A_norm_sum", demo_norm_mode="sum"),  # = baseline
    _variant("abl6_config_B_norm_l2",  demo_norm_mode="l2"),
]


# ============================================================================
# ABL-INJ -- mecanismo de injeção da demografia × demo_feature_set
#
# Todas com use_vision=True e use_demographics=True: a suspeita é que (a) a
# injeção pós-LayerNorm não sobrevive à presença da visão e (b) o colapso da
# head de Task C aparece no demo_vec "extended". As 6 configs cruzam os 3 modos
# de injeção com os 2 feature sets, e TODAS gravam diagnóstico de colapso.
# ============================================================================
ABL_INJ_INJECTION = [
    _variant("inj_current_base",       demo_injection_mode="current",     demo_feature_set="base",     run_collapse_diagnostics=True),
    _variant("inj_renorm_base",        demo_injection_mode="renorm",      demo_feature_set="base",     run_collapse_diagnostics=True),
    _variant("inj_headinject_base",    demo_injection_mode="head_inject", demo_feature_set="base",     run_collapse_diagnostics=True),
    _variant("inj_current_extended",   demo_injection_mode="current",     demo_feature_set="extended", run_collapse_diagnostics=True),
    _variant("inj_renorm_extended",    demo_injection_mode="renorm",      demo_feature_set="extended", run_collapse_diagnostics=True),
    _variant("inj_headinject_extended", demo_injection_mode="head_inject", demo_feature_set="extended", run_collapse_diagnostics=True),
]


# ============================================================================
# ABL-INJ-CTRL -- controle de capacidade do head_inject
#
# `inj_headinject_base` venceu 6/7 métricas com sinal consistente 3/3, mas faz
# DUAS coisas ao mesmo tempo: (1) dá às heads acesso direto ao sinal demográfico
# e (2) aumenta as heads de 1024 para 1536 in_features -- 512 dimensões
# aprendidas a mais. A ABL-INJ não separa as duas.
#
# Este braço é IDÊNTICO a `inj_headinject_base` em tudo -- mesma arquitetura,
# mesmas 1536 dims, mesma capacidade -- exceto que o demo_vec é embaralhado
# entre itens. Isso preserva a capacidade e a distribuição marginal do input, e
# destrói só o vínculo item<->demografia.
#
# Por que embaralhar e não zerar: com demo_vec=0, `demo_proj(0)` é constante e o
# slot concatenado vira um viés fixo -- as 512 dims extras deixam de ser
# utilizáveis, então zerar removeria o sinal E a capacidade, medindo as duas
# coisas juntas de novo.
# ============================================================================
ABL_INJ_CTRL = [
    _variant("injctrl_headinject_shuffled_base",
             demo_injection_mode="head_inject", demo_feature_set="base",
             demo_shuffle=True, run_collapse_diagnostics=True),
]


# Prefixo dos nomes de config de cada ablação -- `runner.summarize_ablation` usa
# isto para achar as linhas do CSV. ABL-1..6 seguem "abl<N>_config_"; ABL-INJ usa
# "inj_" porque seus nomes descrevem o modo, não uma letra de configuração.
ABLATION_NAME_PREFIXES = {
    "ABL-1": "abl1_config_",
    "ABL-2": "abl2_config_",
    "ABL-3": "abl3_config_",
    "ABL-4": "abl4_config_",
    "ABL-5": "abl5_config_",
    "ABL-6": "abl6_config_",
    "ABL-INJ": "inj_",
    "ABL-INJ-CTRL": "injctrl_",
}


ABLATION_SUITE = [
    ("ABL-1", ABL_1_WARMUP),
    ("ABL-2", ABL_2_DEMO_VEC),
    ("ABL-3", ABL_3_TASK_A_WEIGHT),
    ("ABL-4", ABL_4_UNFROZEN_VISION),
    ("ABL-5", ABL_5_MODALITIES),
    ("ABL-6", ABL_6_DEMO_NORM),
    ("ABL-INJ", ABL_INJ_INJECTION),
    ("ABL-INJ-CTRL", ABL_INJ_CTRL),
]


def config_is_baseline_equivalent(config):
    """True se `config` tem exatamente os mesmos hiperparâmetros que a BASELINE
    (independentemente do `name`/`seed`) -- usado para decidir se um config_A
    pode reaproveitar os números da FASE 0 em vez de retreinar."""
    return replace(config, name="", seed=0) == replace(BASELINE, name="", seed=0)


if __name__ == "__main__":
    print("BASELINE:", BASELINE)
    total_variants = sum(len(variants) for _, variants in ABLATION_SUITE)
    print(f"Ablações: {len(ABLATION_SUITE)} | variantes totais (antes de expandir por seed): {total_variants}")
    for ablation_id, variants in ABLATION_SUITE:
        names = [v.name for v in variants]
        print(f"  {ablation_id}: {names}")
