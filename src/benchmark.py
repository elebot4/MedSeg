"""
Model performance benchmarking for medical segmentation inference.

Benchmarks:
1) Forward inference latency over N steps after warmup
2) Sliding-window inference latency over N steps after warmup

Supported runtimes:
- PyTorch checkpoints (standard or quantized)
- ONNX Runtime sessions (existing .onnx model)
"""

import argparse
import itertools
import time
from typing import cast

import numpy as np
import torch

from eval import _sliding_window_inference
from model import UNet


def run_forward_benchmark(engine, model, session, input_name, x, warmup, steps, device, batch_size):
	with torch.no_grad():
		for _ in range(warmup):
			if engine == "pytorch":
				_ = model(x)
			else:
				_ = session.run(None, {input_name: x})

		times_ms = []
		for _ in range(steps):
			if engine == "pytorch" and str(device).startswith("cuda"):
				torch.cuda.synchronize()
			t0 = time.perf_counter()

			if engine == "pytorch":
				_ = model(x)
			else:
				_ = session.run(None, {input_name: x})

			if engine == "pytorch" and str(device).startswith("cuda"):
				torch.cuda.synchronize()
			t1 = time.perf_counter()
			times_ms.append((t1 - t0) * 1000.0)

	mean_ms = float(np.mean(times_ms))
	p50_ms = float(np.percentile(times_ms, 50))
	p90_ms = float(np.percentile(times_ms, 90))
	throughput = (1000.0 / mean_ms) * batch_size
	return mean_ms, p50_ms, p90_ms, throughput


def run_sliding_window_benchmark(
	engine,
	model,
	session,
	input_name,
	sw_input,
	image_shape,
	window_size,
	out_channels,
	warmup,
	steps,
	device,
	batch_size,
):
	steps_size = [max(1, int(w * (1 - 0.25))) for w in window_size]
	starts = []
	for size, win, step in zip(image_shape, window_size, steps_size):
		if win > size:
			raise ValueError(
				f"window_size {window_size} cannot be larger than image_shape {image_shape}"
			)
		idxs = list(range(0, size - win + 1, step))
		if idxs[-1] != size - win:
			idxs.append(size - win)
		starts.append(idxs)

	output_tensor_shape = tuple([int(batch_size), int(out_channels), *[int(v) for v in image_shape]])

	with torch.no_grad():
		for _ in range(warmup):
			if engine == "pytorch":
				_ = _sliding_window_inference(
					model=model,
					batch=sw_input,
					window_size=window_size,
					out_channels=out_channels,
				)
			else:
				sw_output = np.zeros(output_tensor_shape, dtype=np.float32)
				sw_count = np.zeros(image_shape, dtype=np.float32)
				for idxs in itertools.product(*starts):
					slices = tuple(slice(i, i + w) for i, w in zip(idxs, window_size))
					window = sw_input[(slice(None), slice(None), *slices)]
					pred = cast(np.ndarray, session.run(None, {input_name: window})[0])
					exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))
					pred_softmax = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
					sw_output[(slice(None), slice(None), *slices)] += pred_softmax
					sw_count[slices] += 1
				_ = sw_output / np.expand_dims(np.expand_dims(sw_count, axis=0), axis=0)

		times_ms = []
		for _ in range(steps):
			if engine == "pytorch" and str(device).startswith("cuda"):
				torch.cuda.synchronize()
			t0 = time.perf_counter()

			if engine == "pytorch":
				_ = _sliding_window_inference(
					model=model,
					batch=sw_input,
					window_size=window_size,
					out_channels=out_channels,
				)
			else:
				sw_output = np.zeros(output_tensor_shape, dtype=np.float32)
				sw_count = np.zeros(image_shape, dtype=np.float32)
				for idxs in itertools.product(*starts):
					slices = tuple(slice(i, i + w) for i, w in zip(idxs, window_size))
					window = sw_input[(slice(None), slice(None), *slices)]
					pred = cast(np.ndarray, session.run(None, {input_name: window})[0])
					exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))
					pred_softmax = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
					sw_output[(slice(None), slice(None), *slices)] += pred_softmax
					sw_count[slices] += 1
				_ = sw_output / np.expand_dims(np.expand_dims(sw_count, axis=0), axis=0)

			if engine == "pytorch" and str(device).startswith("cuda"):
				torch.cuda.synchronize()
			t1 = time.perf_counter()
			times_ms.append((t1 - t0) * 1000.0)

	mean_ms = float(np.mean(times_ms))
	p50_ms = float(np.percentile(times_ms, 50))
	p90_ms = float(np.percentile(times_ms, 90))
	throughput = (1000.0 / mean_ms) * batch_size
	return mean_ms, p50_ms, p90_ms, throughput


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Benchmark segmentation model inference")
	parser.add_argument(
		"--engine",
		choices=["pytorch", "onnxruntime"],
		default="pytorch",
		help="Inference engine",
	)
	parser.add_argument(
		"--checkpoint",
		type=str,
		default=None,
		help="Path to PyTorch checkpoint (.pt) with model + config",
	)
	parser.add_argument(
		"--onnx_path",
		type=str,
		default=None,
		help="Path to ONNX model file (.onnx)",
	)
	parser.add_argument(
		"--ort_providers",
		type=str,
		default="CPUExecutionProvider",
		help="Comma-separated ONNX Runtime providers",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cuda" if torch.cuda.is_available() else "cpu",
		help="PyTorch device for benchmark",
	)
	parser.add_argument("--batch_size", type=int, default=1, help="Benchmark batch size")
	parser.add_argument("--warmup", type=int, default=10, help="Warmup steps")
	parser.add_argument("--steps", type=int, default=50, help="Measured benchmark steps")
	parser.add_argument(
		"--window_size",
		type=str,
		default=None,
		help="Sliding window size as comma-separated ints, e.g. 64,64,64",
	)
	parser.add_argument(
		"--image_shape",
		type=str,
		default=None,
		help="Input image shape as comma-separated ints, e.g. 128,128,128",
	)
	args = parser.parse_args()

	if args.engine == "pytorch" and args.checkpoint is None:
		raise ValueError("--checkpoint is required for --engine pytorch")
	if args.engine == "onnxruntime" and args.onnx_path is None:
		raise ValueError("--onnx_path is required for --engine onnxruntime")

	model = None
	session = None
	input_name = None
	out_channels = None
	in_channels = None
	input_shape = None

	if args.engine == "pytorch":
		ckpt = torch.load(args.checkpoint, map_location="cpu")
		if not isinstance(ckpt, dict) or "config" not in ckpt or "model" not in ckpt:
			raise ValueError("Checkpoint must be a dict containing 'model' and 'config' keys")

		config = ckpt["config"]
		if not isinstance(config, dict):
			raise ValueError("Checkpoint 'config' must be a dict")

		input_shape = tuple(config["input_shape"])
		in_channels = int(config["in_channels"])
		out_channels = int(config["out_channels"])

		model = (
			UNet(
				input_shape=input_shape,
				in_channels=in_channels,
				out_channels=out_channels,
				num_stages=config["num_stages"],
				base_chs=config["base_chs"],
				norm_type=config["norm_type"],
				act_type=config["act_type"],
				dropout=config["dropout"],
				norm_groups=config["norm_groups"],
				deep_supervision=config["deep_supervision"],
			)
			.to(args.device)
			.eval()
		)
		model.load_state_dict(ckpt["model"])

	if args.image_shape is not None:
		image_shape = tuple(int(x) for x in args.image_shape.split(",") if x.strip())
	else:
		image_shape = input_shape

	if image_shape is None:
		raise ValueError(
			"image_shape is undefined. Provide --image_shape or use a checkpoint with config.input_shape"
		)

	if args.window_size is not None:
		window_size = tuple(int(x) for x in args.window_size.split(",") if x.strip())
	else:
		window_size = image_shape

	if len(window_size) != len(image_shape):
		raise ValueError(f"window_size {window_size} and image_shape {image_shape} must have same rank")

	if args.engine == "onnxruntime":
		import onnxruntime as ort

		providers = [p.strip() for p in args.ort_providers.split(",") if p.strip()]
		session = ort.InferenceSession(args.onnx_path, providers=providers)
		input_meta = session.get_inputs()[0]
		input_name = input_meta.name
		input_dims = input_meta.shape

		if in_channels is None:
			if len(input_dims) >= 2 and isinstance(input_dims[1], int):
				in_channels = int(input_dims[1])
			else:
				in_channels = 1

		if out_channels is None:
			output_dims = session.get_outputs()[0].shape
			if len(output_dims) >= 2 and isinstance(output_dims[1], int):
				out_channels = int(output_dims[1])
			else:
				out_channels = 1

	if in_channels is None or out_channels is None:
		raise ValueError("in_channels and out_channels must be resolved before benchmark")

	in_channels = int(in_channels)
	out_channels = int(out_channels)
	input_tensor_shape = tuple([int(args.batch_size), in_channels, *[int(v) for v in image_shape]])

	print("Benchmark configuration:")
	print(f"- Engine: {args.engine}")
	if args.engine == "pytorch":
		print(f"- Checkpoint: {args.checkpoint}")
		print(f"- Device: {args.device}")
	else:
		print(f"- ONNX model: {args.onnx_path}")
		print(f"- ONNX providers: {args.ort_providers}")
	print(f"- Batch size: {args.batch_size}")
	print(f"- In channels: {in_channels}")
	print(f"- Out channels: {out_channels}")
	print(f"- Image shape: {image_shape}")
	print(f"- Sliding window: {window_size}")
	print(f"- Warmup steps: {args.warmup}")
	print(f"- Measured steps: {args.steps}")

	if args.engine == "pytorch":
		x = torch.randn(input_tensor_shape, device=args.device)
	else:
		x = np.random.standard_normal(size=input_tensor_shape).astype(np.float32)

	print("\n[1/2] Forward benchmark")
	f_mean, f_p50, f_p90, f_tput = run_forward_benchmark(
		engine=args.engine,
		model=model,
		session=session,
		input_name=input_name,
		x=x,
		warmup=args.warmup,
		steps=args.steps,
		device=args.device,
		batch_size=args.batch_size,
	)
	print(f"Forward mean: {f_mean:.3f} ms | p50: {f_p50:.3f} ms | p90: {f_p90:.3f} ms")
	print(f"Forward throughput: {f_tput:.2f} samples/s")

	print("\n[2/2] Sliding-window benchmark")
	if args.engine == "pytorch":
		sw_input = torch.randn(input_tensor_shape, device=args.device)
	else:
		sw_input = np.random.standard_normal(size=input_tensor_shape).astype(np.float32)

	sw_mean, sw_p50, sw_p90, sw_tput = run_sliding_window_benchmark(
		engine=args.engine,
		model=model,
		session=session,
		input_name=input_name,
		sw_input=sw_input,
		image_shape=image_shape,
		window_size=window_size,
		out_channels=out_channels,
		warmup=args.warmup,
		steps=args.steps,
		device=args.device,
		batch_size=args.batch_size,
	)
	print(f"Sliding-window mean: {sw_mean:.3f} ms | p50: {sw_p50:.3f} ms | p90: {sw_p90:.3f} ms")
	print(f"Sliding-window throughput: {sw_tput:.2f} samples/s")
