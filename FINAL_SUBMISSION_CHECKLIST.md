# Final Submission Checklist

This checklist maps the current project state to the submission requirements.

| Requirement | Status | Evidence / remaining work |
| --- | --- | --- |
| Proposal | DONE | Research question, hypothesis, scope, and completed-vs-planned deliverables are stated in `FINAL_REPORT.md`. |
| Introduction and Related Work | DONE | `FINAL_REPORT.md` includes cited discussion of retrieval-augmented robotics, RLBench, robomimic, MimicGen, Uni3D, and PTv3. |
| Method | DONE | The PDF describes data, encoders, metrics, perturbations, projection head, environment, and implementation ownership. |
| Evaluation and Results | DONE | Full 19-task retrieval, subset-8 retrieval, robustness, held-out projection-head evaluation, downstream proxy, tables, figures, and limitations are included. |
| Conclusion | DONE | The report summarizes findings, negative results, limitations, and future work without overstating replay as planning. |
| Implementation | DONE | Code, tests, `requirements.txt`, configuration files, SLURM helpers, and API documentation are present. |
| Reproducibility README | DONE | `README.md`, `ENVIRONMENT_REPRODUCTION.md`, `requirements.txt`, `requirements-cluster.txt`, SLURM scripts, checkpoint hashes, and dataset metadata document reproduction. |
| Real Uni3D backend | DONE | Official checkpoint loaded and passed real-backend smoke validation. |
| Real PTv3 backend | DONE | Pointcept `v1.5.2` runtime and checkpoint passed real-backend forward smoke validation. |
| Downstream planning | PARTIAL | Offline trajectory-transfer proxy is complete. A guarded simulator replay pilot exists, but true learned-planner success is not yet measured and must be described explicitly as a limitation. |
| Action-aware projection head | OPTIONAL | A frozen-backbone trajectory-signature regression pilot is implemented; it should be included only if trained and evaluated with a clean split. |
| Final PDF | DONE | `FINAL_REPORT.pdf` is generated from `FINAL_REPORT.md`; the source builder is `scripts/build_final_report_pdf.py`. |

## Current Result Locations

| Result | Cluster location |
| --- | --- |
| Full retrieval evaluation | `RAG_for_Robotics_outputs/evaluation/retrieval_full/v2_multitask_full` |
| Subset-8 retrieval evaluation | `RAG_for_Robotics_outputs/evaluation/retrieval_all/v2_multitask_subset8` |
| Subset-8 robustness | `RAG_for_Robotics_outputs/evaluation/robustness/v2_multitask_subset8` |
| Downstream transfer proxy | `RAG_for_Robotics_outputs/evaluation/downstream/v2_multitask_full` |

## Final Verification Before PDF Export

1. Confirm all four result directories contain their expected CSV/JSON/Markdown files.
2. Record environment versions, GPU node, checkpoint paths, and checkpoint hashes.
3. Generate figures from the locked CSV files.
4. Write the final report and export it to PDF.
5. Run the README reproduction commands from a clean cluster shell.
