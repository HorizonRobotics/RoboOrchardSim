# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

# INTERNAL

from __future__ import annotations
import dataclasses
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, model_validator


@dataclasses.dataclass(frozen=True)
class OpenPiTransformPipeline:
    """Compiled OpenPI transforms needed for policy inference."""

    data_transforms: Any
    model_transforms: Any
    use_quantile_norm: bool
    action_sequence_keys: Sequence[str] = ("actions",)


class OpenPiModelConfig(BaseModel):
    """User-provided OpenPI model config loaded from YAML."""

    model_type: Literal["pi0", "pi05", "pi0_fast"]
    max_token_len: int = 128
    paligemma_variant: str | None = None
    action_expert_variant: str | None = None

    @model_validator(mode="after")
    def _validate_variants(self) -> "OpenPiModelConfig":
        if self.model_type in {"pi0", "pi05"}:
            if self.paligemma_variant is None:
                raise ValueError("paligemma_variant is required")
            if self.action_expert_variant is None:
                raise ValueError("action_expert_variant is required")
        if self.model_type == "pi0_fast":
            if self.paligemma_variant is not None:
                raise ValueError("paligemma_variant is not used for pi0_fast")
            if self.action_expert_variant is not None:
                raise ValueError(
                    "action_expert_variant is not used for pi0_fast"
                )
        return self


class OpenPiInferenceConfig(BaseModel):
    """User-provided OpenPI inference config loaded from YAML."""

    use_delta_joint_actions: bool = False
    norm_stats_name: str
    default_prompt: str | None = None


def build_openpi_model_config(model_cfg: OpenPiModelConfig) -> Any:
    import openpi.models.pi0_config as pi0_config

    if model_cfg.model_type == "pi0_fast":
        import openpi.models.pi0_fast as pi0_fast

        return pi0_fast.Pi0FASTConfig()

    return pi0_config.Pi0Config(
        pi05=model_cfg.model_type == "pi05",
        max_token_len=model_cfg.max_token_len,
        paligemma_variant=model_cfg.paligemma_variant,
        action_expert_variant=model_cfg.action_expert_variant,
    )


def build_openpi_transform_pipeline(
    inference_cfg: OpenPiInferenceConfig,
    model: Any,
) -> OpenPiTransformPipeline:
    """Create OpenPI transforms for policy inference."""
    import openpi.transforms as transforms

    data_transforms = transforms.Group()
    if inference_cfg.use_delta_joint_actions:
        delta_action_mask = transforms.make_bool_mask(6, -1, 6, -1)
        data_transforms = data_transforms.push(
            inputs=[transforms.DeltaActions(delta_action_mask)],
            outputs=[transforms.AbsoluteActions(delta_action_mask)],
        )
    return OpenPiTransformPipeline(
        data_transforms=data_transforms,
        model_transforms=_model_transforms(model),
        use_quantile_norm=getattr(model.model_type, "name", model.model_type)
        != "PI0",
    )


def _model_transforms(
    model_config: Any,
) -> Any:
    import openpi.models.model as model_module
    import openpi.models.pi0_config as pi0_config
    import openpi.models.tokenizer as tokenizer
    import openpi.transforms as transforms

    match model_config.model_type:
        case model_module.ModelType.PI0:
            return transforms.Group(
                inputs=[
                    transforms.ResizeImages(224, 224),
                    transforms.TokenizePrompt(
                        tokenizer.PaligemmaTokenizer(
                            model_config.max_token_len,
                        ),
                    ),
                    transforms.PadStatesAndActions(model_config.action_dim),
                ],
            )
        case model_module.ModelType.PI05:
            if not isinstance(model_config, pi0_config.Pi0Config):
                raise TypeError("PI05 model config must be Pi0Config")
            return transforms.Group(
                inputs=[
                    transforms.ResizeImages(224, 224),
                    transforms.TokenizePrompt(
                        tokenizer.PaligemmaTokenizer(
                            model_config.max_token_len,
                        ),
                        discrete_state_input=model_config.discrete_state_input,
                    ),
                    transforms.PadStatesAndActions(model_config.action_dim),
                ],
            )
        case model_module.ModelType.PI0_FAST:
            tokenizer_cls = (
                tokenizer.FASTTokenizer
                if model_config.fast_model_tokenizer is None
                else model_config.fast_model_tokenizer
            )
            tokenizer_kwargs = (
                {}
                if model_config.fast_model_tokenizer_kwargs is None
                else model_config.fast_model_tokenizer_kwargs
            )
            return transforms.Group(
                inputs=[
                    transforms.ResizeImages(224, 224),
                    transforms.TokenizeFASTInputs(
                        tokenizer_cls(
                            model_config.max_token_len,
                            **tokenizer_kwargs,
                        ),
                    ),
                ],
                outputs=[
                    transforms.ExtractFASTActions(
                        tokenizer_cls(
                            model_config.max_token_len,
                            **tokenizer_kwargs,
                        ),
                        action_horizon=model_config.action_horizon,
                        action_dim=model_config.action_dim,
                    ),
                ],
            )
    raise ValueError(
        f"Unsupported OpenPI model type: {model_config.model_type}"
    )
