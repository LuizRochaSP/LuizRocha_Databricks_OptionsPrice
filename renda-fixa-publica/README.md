# Renda Fixa Pública Brasil

Motor de renda fixa pública brasileira, de LFT a swaps: títulos públicos federais com as convenções oficiais,
curvas de juros, derivativos de juros e risco de carteira. É um projeto **independente** do pricer de opções
(`ibov-barrier-option-mvp`): não importa nada de lá e tem pacote, testes, notebooks e dashboard próprios.

## Validação

O motor reproduz **centavo a centavo** (6 casas) o PU indicativo da ANBIMA de todos os títulos negociados:
50 de 50 títulos em 18/11/2026 e em 19/11/2026 (LTN, NTN-F, NTN-B, NTN-C e LFT), com o VNA extraído do
próprio arquivo e consistente em todos os vencimentos. O VNA da LFT avançado com a Selic diária do BCB
(SGS 11) bate com o da ANBIMA dentro do truncamento da 6ª casa.

Nos futuros, o PU de ajuste da B3 sai da taxa de ajuste ao centavo em todos os contratos negociados:
43 DI1 e 20 DAP por dia em 18 e 19/11/2026 (317 de 317 ajustes na semana de 13 a 19/11).

| Regra validada | Onde |
| :--- | :--- |
| LTN e NTN-F: PU = soma dos fluxos descontados, truncada na 6ª casa; cupom NTN-F R$ 48,80885 | `titulos/prefixados.py` |
| NTN-B, NTN-B Principal, NTN-C, LFT: cotação truncada na 4ª casa, PU = VNA x cotação truncado na 6ª; cupom NTN-B 2,956301% | `titulos/indexados.py` |
| NTN-C 2031: cupom de 12% a.a. (5,830052% no semestre) em 1º/jan e 1º/jul; conferido no Tesouro Direto (PU a 0,1%) e no IGP-M acumulado | `titulos/indexados.py`, `vna.py` |
| DI1 e DAP: PU = 100.000 / (1 + taxa)^(DU/252), arredondado na 2ª casa | `derivativos/futuros.py` |
| Dias úteis pelo calendário ANBIMA (feriados nacionais, inclusive 20/11 desde 2024) | `calendario.py` |

## Estrutura

```
renda-fixa-publica/
  src/renda_fixa/
    calendario.py         feriados ANBIMA, dias úteis, datas
    arredondamento.py     truncar / arredondar (Decimal)
    titulos/              LTN, NTN-F, NTN-B, NTN-B Principal, NTN-C, LFT: PU, taxa, duration, DV01, convexidade, fluxos
    vna.py                VNA da LFT (Selic), da NTN-B (IPCA) e da NTN-C (IGP-M), com projeção pro rata
    curvas/               curva zero (flat-forward, linear), Nelson-Siegel-Svensson, bootstrap, inflação implícita, z-spread
    derivativos/          DI1, DAP, FRA, ajuste diário, swaps DI x Pré e DI x IPCA (bullet, % do CDI, spread)
    risco.py              carteira, DV01, key-rate DV01, cenários, carry e roll-down, hedge com DI1, VaR histórico, PCA
    mercado.py            "mercado do dia" a partir do arquivo ANBIMA; histórico de curvas; DI1 da B3 x títulos
    dados/                ANBIMA (msAAMMDD.txt), B3 (ajustes de DI1 e DAP), BCB/SGS, Tesouro Direto (desde 2004)
    graficos.py           tema Plotly do projeto (fundo preto, rampa quente)
  tests/                  52 testes unittest, com os arquivos reais da ANBIMA e da B3 de 18 e 19/11/2026
  databricks/
    01_coleta_renda_fixa.py       ANBIMA, B3, BCB e Tesouro Direto com cache; conferência dos VNAs e dos ajustes
    02_precificacao_curvas.py     validação contra a ANBIMA, curvas pré/real/implícita, NSS, DI1 x títulos, CSVs
    03_derivativos_risco.py       DI1, FRA, swaps, carteira, key-rate, cenários, hedge, VaR e PCA
  dashboard/app.py        Streamlit: títulos do dia, curvas (com DI1 da B3), calculadora, DI1 e swaps, risco
  app.yaml, requirements.txt      para Databricks Apps (pasta de origem = renda-fixa-publica)
```

## Uso rápido

```python
from datetime import date
from renda_fixa.dados.anbima import baixar_mercado_secundario
from renda_fixa.mercado import montar_mercado, tabela_titulos, tabela_curvas
from renda_fixa.titulos import NTNB

m = montar_mercado(baixar_mercado_secundario(date(2026, 11, 19), cache_dir="rf_dados"))
tabela_titulos(m)                    # PU motor x ANBIMA, taxa, duration, DV01, resíduo NSS
tabela_curvas(m)                     # pré, real, inflação implícita, termo de 1 ano
NTNB(date(2035, 5, 15)).pu(7.966, m.ref, vna=m.vna("NTN-B"))   # 4239.718523

from renda_fixa.dados.b3 import baixar_ajustes_b3
from renda_fixa.mercado import comparar_di1
comparar_di1(m, baixar_ajustes_b3(m.ref, cache_dir="rf_dados"))  # DI1 x pré dos títulos, spread em bps
```

## Testes e execução

```bash
cd renda-fixa-publica
PYTHONPATH=src python -m unittest discover -s tests -t .
streamlit run dashboard/app.py
```

No Databricks, os notebooks acham o `src` pelo próprio caminho (Git folder) e gravam os dados em
`/Workspace/Users/<usuário>/rf_dados` (widget `pasta_dados`).

## Fontes de dados e limitações

- **ANBIMA** (`msAAMMDD.txt`): oficial, mas só os últimos dias úteis ficam públicos. Rodar o notebook 01
  diariamente (job) acumula o histórico oficial no cache.
- **Tesouro Direto**: histórico longo e público, mas com taxas de 2 casas, da manhã e com o spread do varejo.
  Usado só para variações de curva (VaR e PCA), nunca para validar PU.
- **BCB/SGS**: Selic e CDI diários, IPCA mensal.
- **B3** (`TradeInformationConsolidatedFile`, arquivos.b3.com.br): arquivo público do pregão, com
  histórico de anos; dele saem o PU e a taxa de ajuste de cada DI1 e DAP negociado. Contrato sem negócio
  no dia não aparece. `mercado.comparar_di1` põe a curva de ajustes de DI1 contra a pré dos títulos no mesmo
  prazo (em 19/11/2026: DI1 de +5 a +14 bps acima dos títulos até 2028 e de 10 a 23 bps abaixo de 2029 a
  2035). Depois do último prefixado (jan/2037) a curva dos títulos é extrapolada e a comparação é marcada.
- Só há uma NTN-C em circulação (jan/2031), então não existe curva de IGP-M: o risco usa uma curva plana
  na taxa dela (`MercadoDia.curva_igpm`). O VNA da NTN-C vem do arquivo da ANBIMA; o notebook 01 confere
  contra o IGP-M acumulado (SGS 189) e mostra a projeção de IGP-M implícita.
- A projeção de IPCA da ANBIMA não é baixada: o VNA projetado da NTN-B sai do próprio arquivo de preços
  (`vna_implicito`), e `vna.ipca_projetado_implicito` devolve a projeção embutida.
- Fora do escopo: crédito privado e cupom cambial.
