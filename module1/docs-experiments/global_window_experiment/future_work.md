# Future Work

## If GGTCE supports longer windows

- Container-adaptive window selection based on TMA memory score
- Attention / dilated CNN over long CPU history (parameter-efficient)
- Compare Global long-window vs Hybrid for deployment scenarios without Prophet

## If GGTCE is null

- Non-MSE objectives on Global path (LFHE-style dispersion study)
- Linear AR baseline on raw CPU with long lags (RLLA-style ceiling)
- External datasets with longer train spans

## If subgroup effect is strong

- Train separate models for memory tiers (exploratory)
- Feature: TMA memory score as Global GRU input channel (analogous to HCERL but for memory metadata)

## Orthogonal tracks (unchanged)

- Peak-aware Global GRU (loss weighting)
- Unseen container generalisation
- Hybrid remains on G96 residual window per TMA + HCERL

## Relation to HCERL

HCERL tested **context channels** on Hybrid residuals → null.  
GGTCE tests **temporal depth** on raw CPU for Global → independent question.

Both needed for complete thesis narrative on *where* extra information helps.
