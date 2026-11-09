# Doc 32 — claude-sonnet-5-5 think_off com os system prompts S1–S3 nos memes extremos (treino)

Mesmos memes do doc 31; system = S1, S2 ou S3 (prompts_32.py); user = [imagem, <meme_text>]; effort medium; thinking {"type": "between_tools"}. Voto = maioria dos 3 prompts, sobre todos os memes do grupo (sem maioria de 2 conta como não-YES, como no doc 31). YES Sk = fração de YES entre as respostas parseadas do prompt k. Claude doc 31 = voto com P1–P3 na mensagem user, mesmo modelo e braço.

| grupo | n | humanos | voto YES GLM off | voto YES GLM on | voto YES Claude doc 31 | voto YES Claude S1–S3 | YES S1 | YES S2 | YES S3 | parse | recusas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1Y5N | 388 | 0.167 | 0.719 | 0.510 | 0.286 | 0.157 | 0.129 | 0.175 | 0.155 | 1.000 | 0 |
| ↳ en | 210 | 0.167 | 0.643 | 0.471 | 0.271 | 0.133 | 0.114 | 0.148 | 0.138 | 1.000 | 0 |
| ↳ es | 178 | 0.167 | 0.809 | 0.556 | 0.303 | 0.185 | 0.146 | 0.208 | 0.174 | 1.000 | 0 |
| 3Y3N | 518 | 0.500 | 0.880 | 0.753 | 0.689 | 0.481 | 0.429 | 0.496 | 0.506 | 1.000 | 0 |
| ↳ en | 260 | 0.500 | 0.865 | 0.765 | 0.704 | 0.454 | 0.400 | 0.473 | 0.473 | 1.000 | 0 |
| ↳ es | 258 | 0.500 | 0.895 | 0.740 | 0.674 | 0.508 | 0.457 | 0.519 | 0.539 | 1.000 | 0 |
| 5Y1N | 645 | 0.833 | 0.964 | 0.918 | 0.924 | 0.795 | 0.755 | 0.814 | 0.803 | 1.000 | 0 |
| ↳ en | 310 | 0.833 | 0.961 | 0.916 | 0.919 | 0.816 | 0.794 | 0.823 | 0.810 | 1.000 | 0 |
| ↳ es | 335 | 0.833 | 0.967 | 0.919 | 0.928 | 0.776 | 0.719 | 0.806 | 0.797 | 1.000 | 0 |
