# Slides: o Método 1 (outubro 2026)

Beamer, tema metropolis, 16:9, em português, no molde de `2026-10_progresso`.
Compilar: `pdflatex main.tex` (duas vezes, por causa do roteiro).
Figuras e linhas de tabela geradas: `python3 scripts/deck_figs.py` (lê `data/`).

| arquivo | conteúdo |
|---|---|
| `main.tex` | preâmbulo (o de `2026-10_progresso` + `\inputrows`), título, roteiro |
| `sections/01_operador.tex` | o operador Δ-PoIssoNN, Eq. (1)-(10) e (17), igual à abertura de `2026-10_progresso` |
| `sections/02_metodo1.tex` | o Método 1, Eq. (18)-(22) e FEM (23)-(27), no mesmo estilo; por que -ε log u é um tempo; Método 0 x Método 1 |
| `sections/025_laje.tex` | laje, caso paredes: Método 1 com $g_0 = \tau^*$ em S contra o $g^*$ ótimo para o LAT (só LAT); figuras `laje_gstar_*.png` por `scripts/slab_fig.py` (`data/slab/`) |
| `sections/03_pac2.tex` | as duas tabelas de restrição de fonte (câmara, recuperável) com as linhas do Método 1 (de `reports/investigating_v2/tabelas.tex`, sem o Método 2) |
| `sections/04_lat_primeiro.tex` | resultados de 15/10, DTI004 truth5ti_s1: início A (rede treinada no LAT) x início B (rede ajustada a g*), mapas, erro de LAT, sensibilidade ao início, teste com g conhecido |
| `sections/05_dti004.tex` | DTI004, seis verdades: Método 1 x Camps (tabela por verdade e todos os membros) |
| `sections/tab_*.tex` | linhas de tabela geradas por `scripts/deck_figs.py` |
| `figures/` | `hoje_mapas.png`, `hoje_erro.png`, `hoje_ruido.png`, `dti004_membros.png` (geradas) |
| `data/` | entradas: `stages_A/`, `stages_B/` (train_lat_then_ecg), `noise_*.json` (train_noise_sweep), `knowng_summary.json` (train_known_g), `dti004/` (set1 SUMMARY.md + JSONs por verdade) |

Só os otimizadores mantidos após a varredura aparecem (Adam 1e-4 e 5e-4).
Não há treino do Método 1 na laje: a seção 3 usa só os ventrículos.
