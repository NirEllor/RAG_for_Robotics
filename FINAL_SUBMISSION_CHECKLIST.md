# Final Submission Checklist

This checklist maps the current project state to the submission requirements.

| Requirement | Status | Evidence / remaining work |
| --- | --- | --- |
| Proposal | PARTIAL | The research question, hypotheses, scope, and plan are documented in `3d_retrieval_experiment_infrastructure_spec.md`; a polished proposal section still needs to be included in the final PDF. |
| Introduction and Related Work | TODO | The final PDF still needs a literature discussion with citations for retrieval-augmented planning, RLBench, Uni3D, Pointcept/PTv3, and image-based retrieval. |
| Method | PARTIAL | The implementation specification and API guide document the pipeline; the final PDF still needs a formal method description, preprocessing details, checkpoint details, projection-head definition, and limitations. |
| Evaluation and Results | PARTIAL | Full 19-task retrieval, subset-8 retrieval, and subset-8 robustness outputs exist. The final PDF still needs tables, figures, per-task analysis, and scientific discussion. |
| Conclusion | TODO | Must summarize the result, negative findings, limitations, and future work without overstating the offline proxy. |
| Implementation | DONE | Code, tests, `requirements.txt`, configuration files, SLURM helpers, and API documentation are present. |
| Reproducibility README | PARTIAL | Reproduction scripts exist; `ENVIRONMENT_REPRODUCTION.md` now documents the validated Cluster stack, but the actual Cluster snapshot still needs to be copied into the results package. |
| Real Uni3D backend | DONE | Official checkpoint loaded and passed real-backend smoke validation. |
| Real PTv3 backend | DONE | Pointcept `v1.5.2` runtime and checkpoint passed real-backend forward smoke validation. |
| Downstream planning | PARTIAL | Offline trajectory-transfer proxy is complete. A guarded simulator replay pilot exists, but true learned-planner success is not yet measured and must be described explicitly as a limitation. |
| Action-aware projection head | OPTIONAL | A frozen-backbone trajectory-signature regression pilot is implemented; it should be included only if trained and evaluated with a clean split. |
| Final PDF | TODO | A final PDF document with the required sections has not yet been produced in this repository. |

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
