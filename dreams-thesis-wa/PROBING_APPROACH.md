# Axis 1: Representation Probing - Simple Approach

## Starting Simple: Final Embeddings Only

Instead of extracting all layers immediately, we'll:

1. **Get final embeddings** from DreaMS model (last layer output)
2. **Test probing** on these embeddings for:
   - Physicochemical properties (MW, LogP, TPSA)
   - Functional groups (aromatic, hydroxyl, etc.)
   - Linear vs MLP probes
3. **Validate** with task indicators (kNN, UMAP)
4. **Later**: Expand to per-layer analysis if needed

## Advantages

✅ Simpler setup - one embedding per spectrum  
✅ Faster iteration - test ideas quickly  
✅ Lower storage requirements  
✅ Clearer initial results  
✅ Easy to expand later

## Workflow

```
DreaMS Model → Final Embeddings (shape: N × 1024) → Probes → Results
                                                      ↓
                                              Linear/MLP probes
                                              kNN validation
                                              UMAP viz
```

Once this works, we can add layer-by-layer analysis to see **where** information emerges.
