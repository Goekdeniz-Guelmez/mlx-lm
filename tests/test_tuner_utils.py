# Copyright © 2024 Apple Inc.

import sys
import unittest
from io import StringIO
from unittest.mock import MagicMock

import mlx.nn as nn

from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import calculate_training_steps, print_trainable_parameters


class TestTunerUtils(unittest.TestCase):
    def setUp(self):
        self.capturedOutput = StringIO()
        sys.stdout = self.capturedOutput

    def tearDown(self):
        sys.stdout = sys.__stdout__

    def test_quantized_print_trainable_parameters(self):
        model = MagicMock()
        quantized_linear = MagicMock(spec=nn.QuantizedLinear)
        quantized_linear.weight = MagicMock(size=1e6)
        quantized_linear.bits = 8
        lora_linear = MagicMock(spec=LoRALinear)
        lora_linear.weight = MagicMock(size=2e6)
        lora_linear.parameters.return_value = [lora_linear.weight]

        linear = MagicMock(spec=nn.Linear)
        linear.weight = MagicMock(size=3e6)
        linear.parameters.return_value = [linear.weight]

        model.leaf_modules.return_value = {
            "quantized_linear": quantized_linear,
            "lora_linear": lora_linear,
            "linear": linear,
        }

        model.trainable_parameters.return_value = {
            "layer1.weight": MagicMock(size=1e6),
            "layer3.weight": MagicMock(size=2e6),
        }
        expected_output_8bits = "Trainable parameters: 33.333% (3.000M/9.000M)\n"
        print_trainable_parameters(model)
        self.assertEqual(self.capturedOutput.getvalue(), expected_output_8bits)
        self.capturedOutput.truncate(0)
        self.capturedOutput.seek(0)

        quantized_linear.weight = MagicMock(size=1e6)
        quantized_linear.bits = 4
        expected_output_4bits = "Trainable parameters: 23.077% (3.000M/13.000M)\n"
        print_trainable_parameters(model)
        self.assertEqual(self.capturedOutput.getvalue(), expected_output_4bits)
        self.capturedOutput.truncate(0)
        self.capturedOutput.seek(0)

    def test_print_trainable_parameters(self):
        model = MagicMock()
        linear1 = MagicMock(spec=nn.Linear)
        linear1.weight = MagicMock(size=1e6)
        linear1.parameters.return_value = [linear1.weight]
        linear2 = MagicMock(spec=nn.Linear)
        linear2.weight = MagicMock(size=2e6)
        linear2.parameters.return_value = [linear2.weight]
        lora_linear = MagicMock(spec=LoRALinear)
        lora_linear.weight = MagicMock(size=3e6)
        lora_linear.parameters.return_value = [lora_linear.weight]
        model.leaf_modules.return_value = {
            "linear1": linear1,
            "linear2": linear2,
            "lora_linear": lora_linear,
        }

        model.trainable_parameters.return_value = {
            "layer1.weight": MagicMock(size=1e6),
            "layer3.weight": MagicMock(size=2e6),
        }
        expected_output = "Trainable parameters: 50.000% (3.000M/6.000M)\n"
        print_trainable_parameters(model)
        self.assertEqual(self.capturedOutput.getvalue(), expected_output)


class TestLoraEpochs(unittest.TestCase):
    def test_resolve_train_iterations_with_epochs(self):
        iters, steps_per_epoch = calculate_training_steps(
            num_samples=10, batch_size=4, epochs=2
        )
        self.assertEqual(iters, 4)
        self.assertEqual(steps_per_epoch, 2)

    def test_resolve_train_iterations_rejects_invalid_epochs(self):
        with self.assertRaisesRegex(ValueError, "epochs must be greater than 0"):
            calculate_training_steps(num_samples=100, batch_size=4, epochs=0)

    def test_resolve_train_iterations_rejects_too_small_dataset(self):
        with self.assertRaisesRegex(
            ValueError,
            "Dataset must have at least batch_size=4 examples but only has 3",
        ):
            calculate_training_steps(num_samples=3, batch_size=4, epochs=1)


if __name__ == "__main__":
    unittest.main()
