# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for AutoCast.

This module provides common utility functions used across the AutoCast package.
It includes functions for graph traversal, tensor type inference, model validation,
and mapping setup between nodes, initializers, and value info. These utilities
support the core functionality of model precision conversion.
"""

import logging
from collections import defaultdict

import numpy as np
import onnx

from modelopt.onnx.utils import get_opset_version


def setup_mappings(model: onnx.ModelProto) -> tuple[dict, dict, dict]:
    """Setup and return mappings for model components.

    Args:
        model: ONNX model to create mappings for.

    Returns:
        Tuple containing:
        - value_info_map: Mapping of names to value infos.
        - initializer_map: Mapping of names to initializers.
        - node_to_init_map: Mapping of node names to their initializer inputs.
    """
    value_info_map = {}
    for container in (model.graph.value_info, model.graph.input, model.graph.output):
        value_info_map.update((vi.name, vi) for vi in container)

    initializer_map = {init.name: init for init in model.graph.initializer}

    node_to_init_map = {
        node.name: [
            initializer_map[input_name]
            for input_name in node.input
            if input_name in initializer_map
        ]
        for node in model.graph.node
    }

    return value_info_map, initializer_map, node_to_init_map


def get_consumer_nodes(model: onnx.ModelProto, tensor_name: str) -> list[onnx.NodeProto]:
    """Get all consumer nodes for a given tensor name.

    Args:
        model: The ONNX model to search.
        tensor_name: Name of the tensor to find consumers for.

    Returns:
        list[onnx.NodeProto]: List of nodes that consume the tensor.
    """
    return [n for n in model.graph.node if tensor_name in n.input]


def get_producer_nodes(model: onnx.ModelProto, tensor_name: str) -> list[onnx.NodeProto]:
    """Get all producer nodes for a given tensor name.

    Args:
        model: The ONNX model to search.
        tensor_name: Name of the tensor to find producers for.

    Returns:
        list[onnx.NodeProto]: List of nodes that produce the tensor.
    """
    return [n for n in model.graph.node if tensor_name in n.output]


def get_unique_consumer_node(model: onnx.ModelProto, tensor_name: str) -> onnx.NodeProto:
    """Get a single consumer node and raise exception if there are multiple consumers.

    Args:
        model: The ONNX model to search.
        tensor_name: Name of the tensor to find consumer for.

    Returns:
        onnx.NodeProto: The single consumer node.

    Raises:
        Exception: If there is not exactly one consumer node.
    """
    consumers = get_consumer_nodes(model, tensor_name)
    if len(consumers) != 1:
        raise Exception(f"Expected single consumer for {tensor_name}, found {len(consumers)}")
    return consumers[0]


def get_cast_to_type(cast_node: onnx.NodeProto) -> int:
    """Get the target type from a Cast node.

    Args:
        cast_node: The Cast node to extract type from.

    Returns:
        int: The target type value from the Cast node's 'to' attribute.

    Raises:
        ValueError: If the Cast node does not have a 'to' attribute.
    """
    for attr in cast_node.attribute:
        if attr.name == "to":
            return attr.i
    raise ValueError("Cast node does not have 'to' attribute")


def get_op_types_not_supported_in_low_precision(
    model: onnx.ModelProto,
    min_opset: int,
    low_precision_type: str = "float16",
) -> list[str]:
    """Get a list of ops not supported in low precision for the opset_version = max(model.opset, min_opset).

    An op is considered to be supported if at least one of the inputs may be in low precision.
    Ops where only some of the inputs may be in low precision are considered supported by this function
    and may need special handling. See PrecisionConverter::_should_skip_low_precision_input_conversion.

    Args:
        model: ONNX model.
        min_opset: Minimum opset version.
        low_precision_type: Target precision to reduce to ('float16' or 'bfloat16').

    Returns:
        ops_without_support: List of ops not supported in low precision for the current opset version.
    """
    # Obtain the current model's opset version
    opset_version = max(get_opset_version(model), min_opset)

    # Get all ops precision support information
    precision = "tensor(float16)" if low_precision_type == "float16" else "tensor(bfloat16)"
    model_ops = {n.op_type for n in model.graph.node}
    schemas_dict = defaultdict(dict)
    for schema in onnx.defs.get_all_schemas_with_history():
        if schema.name not in model_ops:
            continue
        float16_supported = False
        for constr in schema.type_constraints:
            if precision in constr.allowed_type_strs:
                float16_supported = True
                break
        schemas_dict[schema.name].update({schema.since_version: float16_supported})

    # Check that all ops are supported in low precision for the current opset version.
    # Otherwise, exclude from conversion.
    ops_without_support = {}
    for op, schema in schemas_dict.items():
        supported_opsets = [k for k, v in schema.items() if v]
        if supported_opsets:
            min_supported_opset = min(supported_opsets)
            if min_supported_opset > opset_version:
                ops_without_support[op] = min_supported_opset
        else:
            ops_without_support[op] = None

    if ops_without_support:
        logging.warning(
            f"{len(ops_without_support)} ops are not supported in '{low_precision_type}' in opset {opset_version}, "
            f"skipping those from conversion. Upgrade the model's opset version as follows to run them in low "
            f" precision: {ops_without_support}."
        )

    return list(ops_without_support.keys())


def load_shape_overrides_from_calibration_data(
    calibration_data: str | dict | None,
) -> dict[str, tuple]:
    """Extract tensor shapes from calibration data.

    Args:
        calibration_data: Path to NPZ/JSON file, or dict with input tensors.

    Returns:
        dict: Mapping of tensor names to their shapes.
    """
    if calibration_data is None:
        return {}

    shape_overrides = {}

    try:
        if isinstance(calibration_data, str):
            if calibration_data.endswith(".npz"):
                data = np.load(calibration_data)
                shape_overrides = {name: data[name].shape for name in data.files}
            elif calibration_data.endswith(".json"):
                from polygraphy.json import load_json

                data_list = load_json(calibration_data, description="input data")
                if data_list and len(data_list) > 0:
                    first_sample = data_list[0]
                    shape_overrides = {name: tensor.shape for name, tensor in first_sample.items()}
            else:
                logging.warning(
                    f"Unknown calibration data format: {calibration_data}. Using default shapes."
                )
        elif isinstance(calibration_data, dict):
            shape_overrides = {name: tensor.shape for name, tensor in calibration_data.items()}
    except Exception as e:
        logging.warning(f"Failed to load shapes from calibration data: {e}. Using default shapes.")

    return shape_overrides


def estimate_model_memory_requirements(
    model: onnx.ModelProto, shape_overrides: dict[str, tuple] | None = None
) -> int:
    """Estimate the memory required to run inference on the model.

    This estimates memory by summing:
    - All initializers (weights, biases, constants)
    - All intermediate tensors (value_info)
    - Model inputs and outputs

    Args:
        model: ONNX model to estimate memory for.
        shape_overrides: Optional dict mapping tensor names to their shapes.
                        Used to resolve dynamic dimensions. If not provided,
                        dynamic dimensions default to 1.

    Returns:
        int: Estimated memory requirement in bytes.

    Note:
        This is a conservative estimate and doesn't account for:
        - ONNX Runtime internal overhead
        - Temporary buffers during operator execution
        - Memory fragmentation
        - Peak memory usage during specific operations
    """
    total_bytes = 0
    shape_overrides = shape_overrides or {}

    # Helper function to get tensor size in bytes
    def get_tensor_size_bytes(tensor_type, tensor_name: str = "") -> int:
        """Calculate tensor size in bytes from ONNX tensor type."""
        if not tensor_type.HasField("tensor_type"):
            return 0

        elem_type = tensor_type.tensor_type.elem_type
        shape = tensor_type.tensor_type.shape

        # Check if we have a shape override from calibration data
        if tensor_name in shape_overrides:
            num_elements = int(np.prod(shape_overrides[tensor_name]))
        else:
            # Get number of elements from the model's shape info
            num_elements = 1
            has_dynamic_dim = False
            for dim in shape.dim:
                if dim.HasField("dim_value"):
                    num_elements *= dim.dim_value
                elif dim.HasField("dim_param") or not dim.WhichOneof("value"):
                    # Dynamic dimension - default to 1
                    num_elements *= 1
                    has_dynamic_dim = True

            if has_dynamic_dim and tensor_name:
                logging.warning(
                    f"Tensor '{tensor_name}' has dynamic dimensions, but calibration data was not provided."
                    f"Using size 1 for unknowns. Computed memory requirements may be underestimated."
                )

        # Get byte size from ONNX type using built-in helper
        try:
            dtype = onnx.helper.tensor_dtype_to_np_dtype(elem_type)
            bytes_per_element = dtype.itemsize
        except (ValueError, KeyError, TypeError):  # Fallback for unsupported types
            bytes_per_element = 1  # default to 1 byte (undefined type)

        return num_elements * bytes_per_element

    # Sum initializers
    for init in model.graph.initializer:
        num_elements = int(np.prod(init.dims)) if init.dims else 1
        try:
            dtype = onnx.helper.tensor_dtype_to_np_dtype(init.data_type)
            total_bytes += num_elements * dtype.itemsize
        except (ValueError, KeyError, TypeError):  # Fallback for unsupported types
            total_bytes += num_elements * 1

    # Sum intermediate tensors (value_info)
    for vi in model.graph.value_info:
        total_bytes += get_tensor_size_bytes(vi.type, vi.name)

    # Sum inputs
    for input_info in model.graph.input:
        # Skip initializers (they're counted above)
        if any(init.name == input_info.name for init in model.graph.initializer):
            continue
        total_bytes += get_tensor_size_bytes(input_info.type, input_info.name)

    # Sum outputs
    for output_info in model.graph.output:
        total_bytes += get_tensor_size_bytes(output_info.type, output_info.name)

    return total_bytes


def format_bytes(bytes_value: int) -> str:
    """Format bytes into human-readable string.

    Args:
        bytes_value: Number of bytes.

    Returns:
        str: Human-readable string (e.g., "1.5 GB").
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"
