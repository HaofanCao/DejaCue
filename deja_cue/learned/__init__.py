"""Research logic for the low-data learned-decoder comparison.

The modules in this package describe the experiment-facing interface shared by
all eight adapted temporal decoders.  Each native decoder plugs into the common
training adapter, consumes the same prepared records, and returns center-width
proposals with foreground logits.
"""

from .checkpoint import (
    CHECKPOINT_KIND,
    build_checkpoint,
    parameter_state_sha256,
    validate_checkpoint,
)
from .batching import build_epoch_batches, summarize_batches
from .protocol import (
    CUBLAS_WORKSPACE_CONFIG,
    DECODER_REGISTRY,
    TRAINING_SEEDS,
    DecoderSpec,
    TrainingConfig,
    configure_deterministic_cuda_workspace,
    decoder_spec,
    make_training_config,
    set_deterministic_seed,
)
from .records import (
    DecodedProposal,
    LearnedRunRecord,
    PreparedBatch,
    build_evaluation_run_records,
    build_positive_run_records,
    build_run_records,
    build_target_masks,
    collate_records,
    decode_proposals,
    reorder_sim_detr_text,
    state_balanced_coordinates,
)
from .development import load_development_histories
from .training import (
    DecoderTrainingAdapter,
    DecoderTrainingComponents,
    EpochSummary,
    LossOutput,
    TrainingResult,
    TrainingStepContext,
    TrainingSummary,
    prepare_training_batch,
    train_and_save_checkpoint,
    train_decoder,
    validate_training_records,
)
from .native import (
    native_training_adapter,
    predict_native_records,
    resolve_native_config,
    verify_vendored_source,
)

__all__ = [
    "CHECKPOINT_KIND",
    "CUBLAS_WORKSPACE_CONFIG",
    "DECODER_REGISTRY",
    "TRAINING_SEEDS",
    "DecodedProposal",
    "DecoderSpec",
    "DecoderTrainingAdapter",
    "DecoderTrainingComponents",
    "EpochSummary",
    "LearnedRunRecord",
    "LossOutput",
    "PreparedBatch",
    "TrainingConfig",
    "TrainingResult",
    "TrainingStepContext",
    "TrainingSummary",
    "build_epoch_batches",
    "build_checkpoint",
    "build_evaluation_run_records",
    "build_positive_run_records",
    "build_run_records",
    "build_target_masks",
    "collate_records",
    "configure_deterministic_cuda_workspace",
    "decode_proposals",
    "decoder_spec",
    "load_development_histories",
    "make_training_config",
    "native_training_adapter",
    "parameter_state_sha256",
    "prepare_training_batch",
    "predict_native_records",
    "reorder_sim_detr_text",
    "resolve_native_config",
    "set_deterministic_seed",
    "state_balanced_coordinates",
    "summarize_batches",
    "train_and_save_checkpoint",
    "train_decoder",
    "validate_checkpoint",
    "validate_training_records",
    "verify_vendored_source",
]
