Checkpoint 5 — Experiment Results Summary
==========================================
Date range: May 5–8, 2026
Dataset: 449 pairs (75 real + 374 augmented), split seed=42, test_ratio=0.2
         225 index pairs, 73 query pairs, 22 CWE groups

================================================================================
1. 20260505_141347_experiment_f59ab1
   Type: Retrieval — embedder comparison
   Config: 6 embedders × HNSW, dim=128, PCA, L2-normalized, graph=G_vuln
   Results (self-retrieval):
     Embedder          hit@1   hit@5   hit@10   MRR    CWE_recall
     wl                0.699   0.795   0.863    0.736  0.439
     gin               0.726   0.877   0.904    0.788  0.443
     netlsd            0.329   0.548   0.630    0.417  0.210
     combined          0.890   0.959   0.973    0.916  0.498  ← best
     codebert_seq      0.753   0.863   0.863    0.793  0.412
     codebert_pattern  0.712   0.836   0.877    0.768  0.415

================================================================================
2. 20260505_190455_combining_ae1579
   Type: Combining strategies — embedding fusion comparison
   Config: 6 fusion strategies for multi-embedder combination
   Results:
     Strategy              hit@1   hit@5   hit@10   MRR    eff_dim
     concat_pca            0.644   0.795   0.849    0.707  2.10
     pca_concat_pca        0.644   0.795   0.849    0.707  2.10
     pca_concat            0.644   0.781   0.836    0.706  1.99
     norm_concat_pca       0.808   0.932   0.959    0.849  30.87  ← best
     4way_concat_pca       0.644   0.795   0.849    0.707  2.10
     4way_norm_concat_pca  0.726   0.863   0.877    0.773  19.26

================================================================================
3. 20260506_133713_experiment_df393d
   Type: Retrieval — subset (gin + combined only)
   Config: Same split/dataset as #1
   Results:
     gin       hit@1=0.726  MRR=0.788  CWE_recall=0.443
     combined  hit@1=0.877  MRR=0.913  CWE_recall=0.501

================================================================================
5. 20260507_171739_patching_a56231                              *** KEY RUN ***
   Type: Patching — GPT-4o, graph v1 prompt, full 73 queries
   Config: model=gpt-4o, prompt_variant=graph, oracle retriever
   Results:
     LLM vulnerability eval:  52.1% fix rate (35 FIXED, 3 PARTIAL, 35 NOT_FIXED)
     ROUGE-1 F1: 0.618  ROUGE-2 F1: 0.538  ROUGE-L F1: 0.549
     BLEU-4: 0.666   Token Jaccard: 0.763   BERTScore F1: 0.728
     Best CWEs:  Use After Free (9/12), Integer Overflow (6/10)
     Worst CWEs: NULL Pointer Deref (3/11), Expired Pointer Deref (0/2)

================================================================================
6. 20260508_124553_patching_c86225
   Type: Patching — GPT-4o, graph_v2 prompt, 41 queries (23 failed CVEs)
   Config: model=gpt-4o, prompt_variant=graph_v2, oracle, CVE-filtered
   Results:
     LLM vulnerability eval:  41.5% fix rate (16 FIXED, 1 PARTIAL, 24 NOT_FIXED)
     ROUGE-1 F1: 0.617  ROUGE-2 F1: 0.542  ROUGE-L F1: 0.549
     vs graph v1 on same CVEs: recovered 12/41 vs 5/41
     Best CWEs:  Use After Free (3/4), Deadlock (2/2)
     Worst CWEs: NULL Pointer Deref (3/9), Expired Pointer Deref (0/2)

================================================================================
7. 20260508_144305_patching_01dd6b
   Type: Patching — GPT-4o, default_v2 prompt, 41 queries (23 failed CVEs)
   Config: model=gpt-4o, prompt_variant=default_v2, oracle, CVE-filtered
   Results:
     LLM vulnerability eval:  26.8% fix rate (8 FIXED, 3 PARTIAL, 30 NOT_FIXED)
     ROUGE-1 F1: 0.625  ROUGE-2 F1: 0.548  ROUGE-L F1: 0.552
     Head-to-head vs graph_v2: graph_v2 wins 10, default_v2 wins 2, tie 27
     Conclusion: graph context helps on hard cases; simpler-is-better disproved

================================================================================
8. gin_codebert_training
   Type: Model training — GIN with CodeBERT node features
   Config: hidden=128, layers=3, dropout=0.3, margin=0.3, lr=0.001
   Results:
     42 epochs (early stopped), train_loss=0.151, val_loss=0.184
     Retrieval: hit@1=0.288, MRR=0.372, CWE_recall=0.212
     Verdict: Poor — CodeBERT features did not help GIN

================================================================================
9. gin_struct_fusion_test
   Type: Index build test (no training/eval)
   Contents: Pre-built HNSW indices for norm_concat_pca and gin_struct fusion
   Results: N/A

================================================================================
10. gin_struct_training
    Type: Model training — GIN-Struct (triplet metric learning, warm start)
    Config: hidden=128, layers=3, dropout=0.2, margin=0.5, lr=0.0005, labels=CVE
    Results:
      94 epochs (early stopped), train_loss=0.012, val_loss=0.018
      Retrieval: hit@1=0.781, MRR=0.826, CWE_recall=0.578
      CVE precision=0.933, recall=0.745, F1=0.808
      Verdict: Strong — best trained model, close to combined embedder

================================================================================

Patching Prompt Variant Comparison (GPT-4o, oracle):
=====================================================
  Run      Prompt      Queries  Fix Rate  ROUGE-L  Notes
  a56231   graph       73       52.1%     0.549    Full query set
  c86225   graph_v2    41       41.5%     0.549    23 failed CVEs only
  01dd6b   default_v2  41       26.8%     0.552    23 failed CVEs only

Note: c86225 and 01dd6b ran on the SAME 41 hard queries (CVEs that failed
under a56231). Fix rates are not directly comparable to a56231's 52.1%.
On these hard cases, graph_v2 significantly outperforms default_v2.

Failure Analysis (graph_v2, 24 NOT_FIXED):
  Wrong function patched:     8/24 (33%)
  Cosmetic changes only:      6/24 (25%)
  Wrong pointer guarded:      4/24 (17%)
  Incomplete fix:             4/24 (17%)
  Root cause misunderstanding: 2/24 (8%)
