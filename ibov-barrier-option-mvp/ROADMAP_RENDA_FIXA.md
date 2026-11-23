# Roadmap: Motor de Renda Fixa Pública Brasil

> Objetivo: transformar o protótipo de `fixed_income/` num motor de renda fixa pública brasileira,
> de LFT a swaps, que reproduza o PU indicativo da ANBIMA e se integre às curvas e ao risco já existentes.
> Branch de trabalho: `feature/renda-fixa-publica` (criada a partir do `main` em 23/11/2026).

## 1. Ponto de partida (main em a121337)

| Bloco | O que existe | Estado |
| :--- | :--- | :--- |
| `di_curve/conventions.py` | Calendário B3, DU/252, vencimentos DI1, taxa efetiva <-> log-DF | Sólido, testado |
| `di_curve/curve.py`, `interpolators.py` | `DiscountCurve`; flat-forward, spline, NSS | Sólido, testado |
| `di_curve/data_loader.py` | DI1/IND do SPRD da B3; ETTJ ANBIMA via CSV | Sólido |
| `di_curve/swap.py` | Swap DI x Pré com fluxos em DU equiespaçados | Funciona, datas aproximadas |
| `di_curve/risk.py` | DV01 paralelo, key-rate DV01, convexidade, PCA | Sólido, só para swap |
| `fixed_income/ltn.py` | PU e duration de LTN | Fórmula certa, sem truncamento |
| `fixed_income/ntnb.py` | PU e duration de NTN-B | Datas de cupom aproximadas (126 DU) |
| `fixed_income/lft.py` | "PU" de LFT | Conceito errado: devolve VNA capitalizado, não PU |
| Notebook `06_coleta_anbima.py` | ETTJ via `pyettj` (PRE, DIC) com fallback mock | Datas fixas, mock silencioso |
| Dashboard | Aba "Curva de Juros (DI)"; card "Swap DI x Pré" em desenvolvimento | Sem títulos |

### Débitos que entram na Fase 1
1. **LFT**: o PU correto é `VNA x cotação/100`, com `cotação = 100 / (1 + taxa)^(DU/252)` (taxa = ágio/deságio sobre a Selic). Hoje a função projeta o VNA para o vencimento.
2. **NTN-B**: cupons nas datas reais (15/mai e 15/nov ou 15/fev e 15/ago), cupom semestral `(1,06)^0,5 - 1`, cotação truncada e PU = VNA projetado x cotação.
3. **Truncamento e arredondamento** das regras do Tesouro e da ANBIMA (taxa com 4 casas, fatores e PU truncados com 6 casas), sem as quais o PU não bate centavo a centavo.
4. **Testes**: não há `unittest` para `fixed_income/`, só notebooks.
5. **Mock silencioso** no notebook 06: um dia sem dados vira curva falsa. Deve falhar ou marcar a qualidade, como no `markets/quality.py` do SPX.

## 2. Princípios

- **Validação primeiro**: cada título só é "pronto" quando reproduz o PU indicativo da ANBIMA (arquivo diário do mercado secundário) com erro <= R$ 0,01 em todos os vencimentos de um dia de teste, e o teste vira `unittest` com valores congelados.
- **Uma API comum por título**: `Bond.cashflows(ref)`, `price_from_yield`, `yield_from_price`, `duration`, `modified_duration`, `dv01`, `convexity`, `price_from_curve(curve)`. LTN, NTN-F, NTN-B, NTN-B Principal e LFT implementam a mesma interface.
- **Reuso**: calendário, `DiscountCurve`, interpoladores e `risk.py` já existem; não duplicar.
- **Dados com cache e falha explícita**, no padrão do `fx_inputs.py` (retries, cache parquet, nenhum mock em produção).
- Toda mudança no motor também vai para a cópia em `dashboard/ibov_barrier/`.
- Notebooks com `%pip install` de todas as libs + `restartPython`. Gráficos Plotly no padrão escuro/quente com `nodarkmode`.

## 3. Fases

### Fase 1: Títulos públicos corretos (núcleo)
Novo pacote `src/ibov_barrier/fixed_income/` reorganizado:

| Módulo | Conteúdo |
| :--- | :--- |
| `conventions.py` | Regras de truncamento/arredondamento, datas de cupom por título, cupom semestral equivalente |
| `bond.py` | Classe base com a API comum e o solver taxa <- PU (Newton com fallback de bisseção) |
| `ltn.py` | LTN: prefixado sem cupom, VN R$ 1.000 |
| `ntnf.py` | **Novo**. NTN-F: prefixado, cupom 10% a.a. pago em 1/jan e 1/jul |
| `ntnb.py` | NTN-B (cupom 6% a.a.) e NTN-B Principal; VNA projetado com IPCA projetado (pro rata entre os dias 15) |
| `lft.py` | LFT: VNA atualizado pela Selic diária, cotação por ágio/deságio, PU |
| `vna.py` | VNA de NTN-B (IPCA, SGS 433, + projeção ANBIMA) e de LFT (Selic, SGS 11), com cache |

Entregas: 5 títulos com PU, taxa implícita, duration de Macaulay e modificada, DV01, convexidade e juros acumulados. Testes com exemplos da metodologia ANBIMA e de um dia real.

### Fase 2: Dados de mercado
- `fixed_income/data_anbima.py`: taxas e PUs indicativos dos títulos públicos (arquivo diário do mercado secundário da ANBIMA), VNA e projeções de IPCA/IGP-M.
- ETTJ via `pyettj` com cache e flag de qualidade, sem mock.
- Opcional: preços do Tesouro Direto (transparência do Tesouro) para comparar varejo vs indicativo.
- Notebook `11_coleta_renda_fixa.py`: baixa o dia, grava parquet em `rf_dados/` e um relatório de cobertura.

### Fase 3: Curvas
- **ETTJ prefixada**: bootstrap a partir de LTN e NTN-F (e opcionalmente DI1, que já existe), comparando com a ETTJ PRE ANBIMA.
- **ETTJ IPCA (juro real)**: bootstrap das NTN-B, com NSS para suavizar.
- **Inflação implícita (breakeven)**: `(1 + pré) / (1 + real) - 1` por vértice.
- **Spread título x DI** por vencimento (rico/barato).
- Validação contra a ETTJ ANBIMA (`validation.compare_with_anbima` já existe).

### Fase 4: Derivativos de juros
| Instrumento | Conteúdo |
| :--- | :--- |
| Futuro DI1 | PU = 100.000 / (1 + taxa)^(DU/252), DV01 por contrato, ajuste diário |
| FRA de DI | Taxa a termo entre dois vértices; estrutura de FRA (OC1/DI1) |
| Swap DI x Pré | Refatorar `swap.py`: datas reais de pagamento, CDI% e spread, MtM a partir de uma data de início no passado (CDI acumulado) |
| Swap DI x IPCA | Ponta IPCA + cupom contra % do CDI, usando a ETTJ real |
| Futuro DAP | Cupom de IPCA da B3; hedge de NTN-B |
| Opcional | Opções de IDI e de DI1 (Black sobre a taxa a termo), a ligar com o núcleo de opções |

### Fase 5: Risco e análise de carteira
- Carteira mista (títulos + DI1 + swaps): DV01 total, key-rate DV01 por vértice, convexidade.
- Hedge: quantos DI1 neutralizam uma LTN/NTN-F; DAP x NTN-B.
- Cenários: choque paralelo, inclinação, curvatura e cenários PCA (reusar `risk.pca_decomposition`); integrar com `04_cenarios_stress`.
- Carry e roll-down por título; atribuição de P&L (carry, curva, spread).
- VaR histórico da carteira de renda fixa.

### Fase 6: Notebook de produção e dashboard
- Notebook `12_precificacao_renda_fixa.py`: precifica todos os títulos do dia, compara com a ANBIMA (tabela de erros), curvas pré/real/implícita e risco. Grava um bundle `RESULTS_renda_fixa_*.csv` (formato de `results_bundle.py`).
- Dashboard: aba "Renda Fixa Pública" com calculadora (taxa <-> PU), tabela dos títulos do dia, curvas e breakeven, swap DI x Pré funcional e risco de carteira.
- Deploy só quando o usuário decidir.

## 4. Ordem e critérios de aceite

| # | Fase | Aceite |
| :--- | :--- | :--- |
| 1 | Títulos | 5 títulos com `unittest`; exemplos da metodologia batem exatamente |
| 2 | Dados | Notebook 11 roda no Databricks; cache e flag de qualidade |
| 3 | Validação real | PU de todos os títulos do dia com erro <= R$ 0,01 vs ANBIMA |
| 4 | Curvas | Pré e real a menos de 1 bp da ETTJ ANBIMA nos vértices líquidos |
| 5 | Derivativos | DI1, FRA, swaps DI x Pré e DI x IPCA, DAP; MtM de swap em andamento |
| 6 | Risco | DV01/key-rate de carteira mista; hedge com DI1 |
| 7 | Produção | Notebook 12 + aba do dashboard + suíte completa verde |

## 5. Fora de escopo por ora
Crédito privado (debêntures, CRI/CRA, spread de crédito), NTN-C/IGP-M e cupom cambial (DDI/FRC). Podem virar uma Fase 7.
