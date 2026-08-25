from action_retrieval.data.schema import EpisodeRecord


def test_episode_record_to_dict_serializes_sequences_as_lists():
    record = EpisodeRecord(
        dataset_version="v1_reach_target",
        task_name="reach_target",
        episode_id="episode0",
        variation_id=0,
        seed=42,
        split="train",
        success=True,
        num_observations=70,
        snapshot_policy="initial",
        coordinate_frame="world",
        source_kind="saved_demo",
        source_root="/tmp/source",
        language_descriptions=("reach the target", "move to the target"),
        camera_names=("front",),
        observation_path="episodes/reach_target/episode0/observation.npz",
        trajectory_path="episodes/reach_target/episode0/trajectory.npz",
        metadata_path="episodes/reach_target/episode0/metadata.json",
    )

    payload = record.to_dict()

    assert payload["language_descriptions"] == ["reach the target", "move to the target"]
    assert payload["camera_names"] == ["front"]
    assert payload["split"] == "train"
