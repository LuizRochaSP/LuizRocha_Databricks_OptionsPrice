# Renda Fixa Pública Brasil

Motor de renda fixa pública brasileira, de LFT a swaps: títulos públicos federais com as convenções oficiais,
curvas de juros, derivativos de juros e risco de carteira. É um projeto **independente** do pricer de opções
(`ibov-barrier-option-mvp`): não importa nada de lá e tem pacote, testes, notebooks e dashboard próprios.

## Validação

O motor reproduz **centavo a centavo** (6 casas) o PU indicativo da ANBIMA de todos os títulos negociados:
49 de 49 títulos em 18/11/2026 e em 19/11/2026 (LTN, NTN-F, NTN-B e LFT), com o VNA extraído do próprio
arquivo e consistente em todos os vencimentos. O VNA da LFT avançado com a Selic diária do BCB
(SGS 11) bate com o da ANBIMA dentro do truncamento da 6ª casa.

| Regra validada | Onde |
| :--- | :--- |
| LTN e NTN-F: PU = soma dos fluxos descontados, truncada na 6ª casa; cupom NTN-F R$ 48,80885 | `titulos/prefixados.py` |
| NTN-B, NTN-B Principal, LFT: cotação truncada na 4ª casa, PU = VNA x cotação truncado na 6ª; cupom NTN-B 2,956301% | `titulos/indexados.py` |
| Dias úteis pelo calendário ANBIMA (feriados nacionais, inclusive 20/11 desde 2024) | `calendario.py` |

## Estrutura

```
renda-fixa-publica/
  src/renda_fixa/
    calendario.py         feriados ANBIMA, dias úteis, datas
    arredondamento.py     truncar / arredondar (Decimal)
    titulos/              LTN, NTN-F, NTN-B, NTN-B Principal, LFT: PU, taxa, duration, DV01, convexidade, fluxos
    vna.py                VNA da LFT (Selic) e da NTN-B (IPCA, projeção pro rata)
    curvas/               curva zero (flat-forward, linear), Nelson-Siegel-Svensson, bootstrap, inflação implícita, z-spread
    derivativos/          DI1, DAP, FRA, ajuste diário, swaps DI x Pré e DI x IPCA (bullet, % do CDI, spread)
    risco.py              carteira, DV01, key-rate DV01, cenários, carry e roll-down, hedge com DI1, VaR histórico, PCA
    mercado.py            "mercado do dia" a partir do arquivo ANBIMA; histórico de curvas
    dados/                ANBIMA (msAAMMDD.txt), BCB/SGS, Tesouro Direto (histórico desde 2004)
    graficos.py           tema Plotly do projeto (fundo preto, rampa quente)
  tests/                  41 testes unittest, com os arquivos ANBIMA reais de 18 e 19/11/2026
  databricks/
    01_coleta_renda_fixa.py       ANBIMA, BCB e Tesouro Direto com cache; conferência do VNA
    02_precificacao_curvas.py     validação contra a ANBIMA, curvas pré/real/implícita, NSS, CSVs de resultado
    03_derivativos_risco.py       DI1, FRA, swaps, carteira, key-rate, cenários, hedge, VaR e PCA
  dashboard/app.py        Streamlit: títulos do dia, curvas, calculadora, DI1 e swaps, risco de carteira
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
- DI1 e DAP são precificados pela curva dos títulos; o projeto não depende de dados da B3. Uma curva de
  ajustes de DI1 pode ser montada com `derivativos.curva_di1`.
- A projeção de IPCA da ANBIMA não é baixada: o VNA projetado da NTN-B sai do próprio arquivo de preços
  (`vna_implicito`), e `vna.ipca_projetado_implicito` devolve a projeção embutida.
- Fora do escopo: crédito privado, NTN-C (IGP-M) e cupom cambial.
