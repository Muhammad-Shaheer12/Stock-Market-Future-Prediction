from __future__ import annotations
from typing import Dict
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def up_down_baskets(df_by_symbol: Dict[str, pd.DataFrame], horizon: int = 1):
    # Build transactions where each day basket contains symbols that moved up
    aligned = None
    ups: Dict[str, pd.Series] = {}
    for sym, df in df_by_symbol.items():
        future = df["adj_close"].shift(-horizon)
        up = (future > df["adj_close"]).astype(int)
        ups[sym] = up
        if aligned is None:
            aligned = df["date"]
    # Create a binary occurrence matrix for symbols across dates
    mat = pd.DataFrame(ups).dropna().astype(int)
    # Frequent itemsets of ups
    itemsets = apriori(mat, min_support=0.1, use_colnames=True)
    rules = association_rules(itemsets, metric="confidence", min_threshold=0.5)
    rules = rules.sort_values(["lift", "confidence"], ascending=False)
    out = []
    for _, r in rules.iterrows():
        out.append(
            {
                "antecedents": list(r["antecedents"]),
                "consequents": list(r["consequents"]),
                "support": float(r["support"]),
                "confidence": float(r["confidence"]),
                "lift": float(r["lift"]),
            }
        )
    return out[:20]
