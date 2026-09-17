"""Configurable seq2seq LSTM for multi-week fantasy-point forecasts."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from itertools import product
from typing import Any, Iterable, Mapping

import tensorflow as tf
from absl import logging as absl_logging

absl_logging.set_verbosity(absl_logging.ERROR)
"""Configurable TensorFlow/Keras seq2seq LSTM for fantasy forecasts."""

class Seq2SeqLSTM(tf.keras.Model):
    """Encode player history and decode a configurable forecast horizon."""
    def __init__(
        self,
        numeric_feature_count: int,
        categorical_cardinalities: Mapping[str, int],
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        data_config = config["data"]
        model_config = config["model"]
        self.huber_delta = float(config["training"].get("huber_delta", 1.0))
        self.horizon = int(data_config["look_forward"])
        self.use_played_mask = bool(model_config.get("use_played_mask", True))
        categorical_features = data_config.get("categorical_features", [])
        embedding_dims = model_config.get("categorical_embedding_dims", {})
        self.embeddings = {
            feature: tf.keras.layers.Embedding(
                input_dim=int(categorical_cardinalities[feature]),
                output_dim=int(embedding_dims.get(feature, 8)),
                name=f"{feature}_embedding",
            )
            for feature in categorical_features
        }
        encoder_sizes = [int(size) for size in model_config.get("encoder_hidden_sizes", [model_config.get("hidden_size", 128)])]
        decoder_sizes = [int(size) for size in model_config.get("decoder_hidden_sizes", [model_config.get("hidden_size", 128)])]
        dropout = float(model_config.get("dropout", 0.0))
        bidirectional = bool(model_config.get("bidirectional", False))
        embedding_size = sum(layer.output_dim for layer in self.embeddings.values())
        encoder_input_size = numeric_feature_count + embedding_size + int(self.use_played_mask)
        self.encoder = []
        current_size = encoder_input_size
        for index, hidden_size in enumerate(encoder_sizes):
            layer = tf.keras.layers.LSTM(hidden_size, return_sequences=True, name=f"encoder_lstm_{index + 1}")
            if bidirectional:
                layer = tf.keras.layers.Bidirectional(layer, name=f"encoder_bidirectional_{index + 1}")
            self.encoder.append(layer)
            current_size = hidden_size * (2 if bidirectional else 1)
        self.encoder_dropout = tf.keras.layers.Dropout(dropout)
        horizon_embedding_dim = int(model_config.get("decoder_horizon_embedding_dim", 16))
        self.horizon_embeddings = tf.keras.layers.Embedding(self.horizon, horizon_embedding_dim, name="horizon_embedding")
        self.decoder = []
        current_size = current_size + horizon_embedding_dim
        for index, hidden_size in enumerate(decoder_sizes):
            self.decoder.append(tf.keras.layers.LSTM(hidden_size, return_sequences=True, name=f"decoder_lstm_{index + 1}"))
            current_size = hidden_size
        self.decoder_dropout = tf.keras.layers.Dropout(dropout)
        self.output_layer = tf.keras.layers.Dense(1, name="forecast")
        activation_name = model_config.get("output_activation", "identity").lower()
        activations = {"identity": tf.keras.activations.linear, "relu": tf.keras.activations.relu, "softplus": tf.keras.activations.softplus}
        if activation_name not in activations:
            raise ValueError(f"Unsupported output activation: {activation_name}")
        self.output_activation = activations[activation_name]

    def compile(self, optimizer: tf.keras.optimizers.Optimizer, loss: tf.keras.losses.Loss, **kwargs: Any) -> None:
        super().compile(optimizer=optimizer, loss=loss, weighted_metrics=[], **kwargs)
        self.masked_loss = loss

    def _masked_objective(self, targets: tf.Tensor, predictions: tf.Tensor, weights: tf.Tensor | None) -> tf.Tensor:
        error = targets - predictions
        if isinstance(self.masked_loss, tf.keras.losses.Huber):
            delta = tf.cast(self.huber_delta, predictions.dtype)
            absolute_error = tf.abs(error)
            quadratic = tf.minimum(absolute_error, delta)
            per_step_loss = 0.5 * tf.square(quadratic) + delta * (absolute_error - quadratic)
        elif isinstance(self.masked_loss, tf.keras.losses.MeanSquaredError):
            per_step_loss = tf.square(error)
        elif isinstance(self.masked_loss, tf.keras.losses.MeanAbsoluteError):
            per_step_loss = tf.abs(error)
        else:
            per_step_loss = self.masked_loss(targets, predictions)
            if per_step_loss.shape.rank == 1:
                per_step_loss = per_step_loss[:, tf.newaxis]
        if weights is None:
            return tf.reduce_mean(per_step_loss)
        weights = tf.cast(weights, per_step_loss.dtype)
        return tf.reduce_sum(per_step_loss * weights) / tf.maximum(tf.reduce_sum(weights), 1.0)

    def train_step(self, data: tuple[Any, Any, Any]) -> dict[str, tf.Tensor]:
        inputs, targets, weights = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self._masked_objective(targets, predictions, weights)
            if self.losses:
                loss += tf.add_n(self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}

    def test_step(self, data: tuple[Any, Any, Any]) -> dict[str, tf.Tensor]:
        inputs, targets, weights = tf.keras.utils.unpack_x_y_sample_weight(data)
        predictions = self(inputs, training=False)
        return {"loss": self._masked_objective(targets, predictions, weights)}

    def call(
        self,
        inputs: Mapping[str, tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        """Return predictions shaped ``(batch, look_forward)``."""
        numeric_inputs = inputs["numeric"]
        categorical_inputs = inputs["categorical"]
        if categorical_inputs.shape.rank != 3 or numeric_inputs.shape.rank != 3:
            raise ValueError("Inputs must be shaped (batch, time, features)")
        embedded = [layer(categorical_inputs[:, :, index]) for index, layer in enumerate(self.embeddings.values())]
        encoder_inputs = [numeric_inputs, *embedded]
        if self.use_played_mask:
            played_mask = inputs.get("played_mask")
            if played_mask is None:
                played_mask = tf.ones(tf.shape(numeric_inputs)[:2], dtype=numeric_inputs.dtype)
            encoder_inputs.append(tf.expand_dims(tf.cast(played_mask, numeric_inputs.dtype), axis=-1))
        encoder_output = tf.concat(encoder_inputs, axis=-1)
        for index, layer in enumerate(self.encoder):
            encoder_output = layer(encoder_output, training=training)
            if index < len(self.encoder) - 1:
                encoder_output = self.encoder_dropout(encoder_output, training=training)
        context = encoder_output[:, -1:, :]
        context = tf.repeat(context, repeats=self.horizon, axis=1)
        horizon_indices = tf.range(self.horizon)[tf.newaxis, :]
        horizon_indices = tf.tile(horizon_indices, [tf.shape(numeric_inputs)[0], 1])
        decoder_output = tf.concat([context, self.horizon_embeddings(horizon_indices)], axis=-1)
        for index, layer in enumerate(self.decoder):
            decoder_output = layer(decoder_output, training=training)
            if index < len(self.decoder) - 1:
                decoder_output = self.decoder_dropout(decoder_output, training=training)
        return tf.squeeze(self.output_activation(self.output_layer(decoder_output)), axis=-1)


class WideDeepForecastModel(tf.keras.Model):
    """Late-fusion network for numeric and categorical features, outputting all horizons at once."""

    def __init__(
        self,
        numeric_feature_count: int,
        categorical_cardinalities: Mapping[str, int],
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        model_config = config["model"]
        data_config = config["data"]
        self.huber_delta = float(config["training"].get("huber_delta", 1.0))
        self.horizon = int(data_config["look_forward"])
        self.use_played_mask = bool(model_config.get("use_played_mask", True))
        categorical_features = data_config.get("categorical_features", [])
        embedding_dims = model_config.get("categorical_embedding_dims", {})
        self.embeddings = {
            feature: tf.keras.layers.Embedding(
                input_dim=int(categorical_cardinalities[feature]),
                output_dim=int(embedding_dims.get(feature, 8)),
                name=f"{feature}_embedding",
            )
            for feature in categorical_features
        }
        numeric_units = int(model_config.get("numeric_branch_width", 128))
        categorical_units = int(model_config.get("categorical_branch_width", 64))
        dropout = float(model_config.get("dropout", 0.0))
        self.numeric_branch = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(numeric_units, activation="relu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(numeric_units // 2, activation="relu"),
            ],
            name="numeric_branch",
        )
        self.categorical_branch = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(categorical_units, activation="relu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(categorical_units // 2, activation="relu"),
            ],
            name="categorical_branch",
        )
        self.fusion = tf.keras.layers.Dense(max(numeric_units // 2, categorical_units // 2), activation="relu")
        self.output_layer = tf.keras.layers.Dense(self.horizon, name="forecast")
        activation_name = model_config.get("output_activation", "identity").lower()
        activations = {"identity": tf.keras.activations.linear, "relu": tf.keras.activations.relu, "softplus": tf.keras.activations.softplus}
        if activation_name not in activations:
            raise ValueError(f"Unsupported output activation: {activation_name}")
        self.output_activation = activations[activation_name]
        self.masked_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)

    def compile(self, optimizer: tf.keras.optimizers.Optimizer, loss: tf.keras.losses.Loss, **kwargs: Any) -> None:
        super().compile(optimizer=optimizer, loss=loss, weighted_metrics=[], **kwargs)
        self.masked_loss = loss

    def _masked_objective(self, targets: tf.Tensor, predictions: tf.Tensor, weights: tf.Tensor | None) -> tf.Tensor:
        error = targets - predictions
        loss_obj = self.masked_loss
        if isinstance(loss_obj, tf.keras.losses.Huber):
            delta = tf.cast(self.huber_delta, predictions.dtype)
            absolute_error = tf.abs(error)
            quadratic = tf.minimum(absolute_error, delta)
            per_step_loss = 0.5 * tf.square(quadratic) + delta * (absolute_error - quadratic)
        elif isinstance(loss_obj, tf.keras.losses.MeanSquaredError):
            per_step_loss = tf.square(error)
        elif isinstance(loss_obj, tf.keras.losses.MeanAbsoluteError):
            per_step_loss = tf.abs(error)
        else:
            per_step_loss = loss_obj(targets, predictions)
            if per_step_loss.shape.rank == 1:
                per_step_loss = per_step_loss[:, tf.newaxis]
        if weights is None:
            return tf.reduce_mean(per_step_loss)
        weights = tf.cast(weights, per_step_loss.dtype)
        return tf.reduce_sum(per_step_loss * weights) / tf.maximum(tf.reduce_sum(weights), 1.0)

    def train_step(self, data: tuple[Any, Any, Any]) -> dict[str, tf.Tensor]:
        inputs, targets, weights = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self._masked_objective(targets, predictions, weights)
            if self.losses:
                loss += tf.add_n(self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}

    def test_step(self, data: tuple[Any, Any, Any]) -> dict[str, tf.Tensor]:
        inputs, targets, weights = tf.keras.utils.unpack_x_y_sample_weight(data)
        predictions = self(inputs, training=False)
        return {"loss": self._masked_objective(targets, predictions, weights)}

    def call(
        self,
        inputs: Mapping[str, tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        numeric_inputs = inputs["numeric"]
        categorical_inputs = inputs["categorical"]
        if categorical_inputs.shape.rank != 3 or numeric_inputs.shape.rank != 3:
            raise ValueError("Inputs must be shaped (batch, time, features)")
        embedded = [layer(categorical_inputs[:, -1, index]) for index, layer in enumerate(self.embeddings.values())]
        if self.use_played_mask:
            played_mask = inputs.get("played_mask")
            if played_mask is None:
                played_mask = tf.ones(tf.shape(numeric_inputs)[:2], dtype=numeric_inputs.dtype)
            numeric_inputs = tf.concat(
                [numeric_inputs[:, -1, :], tf.reduce_mean(tf.cast(played_mask, numeric_inputs.dtype), axis=1, keepdims=True)],
                axis=-1,
            )
        else:
            numeric_inputs = numeric_inputs[:, -1, :]
        if embedded:
            cat_vector = tf.concat(embedded, axis=-1)
        else:
            cat_vector = tf.zeros((tf.shape(numeric_inputs)[0], 0), dtype=numeric_inputs.dtype)
        numeric_repr = self.numeric_branch(numeric_inputs)
        categorical_repr = self.categorical_branch(cat_vector)
        fused = tf.concat([numeric_repr, categorical_repr], axis=-1)
        return self.output_activation(self.output_layer(self.fusion(fused)))


def build_model(
    config: Mapping[str, Any],
    numeric_feature_count: int,
    categorical_cardinalities: Mapping[str, int],
) -> tf.keras.Model:
    """Construct a model from the loaded YAML configuration."""
    architecture = str(config.get("model", {}).get("architecture", "seq2seq_lstm")).lower()
    if architecture == "seq2seq_lstm":
        return Seq2SeqLSTM(numeric_feature_count, categorical_cardinalities, config)
    if architecture in {"wide_deep_mlp", "hybrid_fusion", "late_fusion_mlp"}:
        return WideDeepForecastModel(numeric_feature_count, categorical_cardinalities, config)
    raise ValueError(f"Unsupported model architecture: {architecture}")


def build_optimizer(model: tf.keras.Model, config: Mapping[str, Any]) -> tf.keras.optimizers.Optimizer:
    """Construct the configured Keras optimizer."""
    training_config = config["training"]
    optimizer_name = training_config.get("optimizer", "AdamW").lower()
    optimizer_namespace = tf.keras.optimizers
    if training_config.get("optimizer_api", "standard").lower() == "legacy":
        optimizer_namespace = getattr(tf.keras.optimizers, "legacy", optimizer_namespace)
    optimizer_types = {
        "adam": getattr(optimizer_namespace, "Adam", None),
        "adamw": getattr(optimizer_namespace, "AdamW", None),
        "sgd": getattr(optimizer_namespace, "SGD", None),
    }
    if optimizer_name not in optimizer_types:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    if optimizer_types[optimizer_name] is None:
        raise ValueError(f"Optimizer {optimizer_name} is unavailable in the selected Keras optimizer API")
    kwargs = {
        "learning_rate": float(training_config["learning_rate"]),
        "weight_decay": float(training_config.get("weight_decay", 0.0)),
    }
    if optimizer_name == "sgd":
        kwargs["momentum"] = float(training_config.get("momentum", 0.0))
    return optimizer_types[optimizer_name](**kwargs)


def build_loss(config: Mapping[str, Any]) -> tf.keras.losses.Loss:
    """Construct the configured point-forecast loss."""
    loss_name = config["training"].get("loss", "huber").lower()
    losses = {"huber": tf.keras.losses.Huber, "mse": tf.keras.losses.MeanSquaredError, "mae": tf.keras.losses.MeanAbsoluteError}
    if loss_name not in losses:
        raise ValueError(f"Unsupported loss: {loss_name}")
    return losses[loss_name](reduction=tf.keras.losses.Reduction.NONE)


def _range_values(specification: Mapping[str, Any]) -> list[Any]:
    """Expand an inclusive typed start/stop/step specification."""
    value_type = specification["type"]
    if value_type == "choices":
        return list(specification["values"])
    if value_type not in {"int", "float"}:
        raise ValueError(f"Unsupported search-space type: {value_type}")
    start = Decimal(str(specification["start"]))
    stop = Decimal(str(specification["stop"]))
    step = Decimal(str(specification["step"]))
    if step <= 0:
        raise ValueError("Search-space step must be positive")
    values = []
    current = start
    while current <= stop:
        values.append(int(current) if value_type == "int" else float(current))
        current += step
    return values


def _set_dotted_value(config: dict[str, Any], dotted_name: str, value: Any) -> None:
    target = config
    path = dotted_name.split(".")
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def expand_search_space(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return independent full-config combinations from the YAML search space."""
    search_space = config.get("search_space", {})
    names = list(search_space)
    values = [_range_values(search_space[name]) for name in names]
    combinations = []
    for selection in product(*values):
        candidate = deepcopy(dict(config))
        for name, value in zip(names, selection):
            _set_dotted_value(candidate, name, value)
        combinations.append(candidate)
    return combinations


