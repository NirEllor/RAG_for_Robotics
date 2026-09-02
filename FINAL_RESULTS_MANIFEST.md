# Final Results Manifest

This file records the current presentation-ready result locations. It is kept
separate from raw data and checkpoints so the final package can be reproduced.

| Component | Status | Location |
| --- | --- | --- |
| Full 19-task dataset | Complete | `RAG_for_Robotics_data/processed/v2_multitask_full` |
| Full retrieval evaluation | Complete | `RAG_for_Robotics_outputs/evaluation/retrieval_full/v2_multitask_full` |
| Subset-8 retrieval evaluation | Complete | `RAG_for_Robotics_outputs/evaluation/retrieval_all/v2_multitask_subset8` |
| Subset-8 robustness evaluation | Complete | `RAG_for_Robotics_outputs/evaluation/robustness/v2_multitask_subset8` |
| Downstream trajectory-transfer proxy | Complete | `RAG_for_Robotics_outputs/evaluation/downstream/v2_multitask_full` |
| Real Uni3D backend | Validated | Clean Pointcept environment + official checkpoint |
| Real PTv3 backend | Validated | Pointcept `v1.5.2` + official checkpoint |

The downstream proxy is explicitly not a simulator success-rate experiment.
It measures whether retrieval transfers a task-compatible trajectory and
whether the stored trajectory structure is usable by a downstream consumer.
