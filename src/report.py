"""
Borrowed from nanochat and adapted for this medical segmentation project.
Utilities for lightweight experiment report logging and generation.
"""

import argparse
import datetime
import os
import platform
import shutil
import socket
import subprocess

import psutil
import torch

SECTION_ORDER = [
    "data-preparation.md",
    "training.md",
    "validation.md",
    "evaluation.md",
    "quantization.md",
    "export.md",
    "segmentation-summary.md",
]


def run_command(cmd):
    """Run a shell command and return stdout, empty string, or None on failure."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None

    if result.stdout.strip():
        return result.stdout.strip()
    if result.returncode == 0:
        return ""
    return None


def get_git_info():
    """Get git branch/commit status for reproducibility."""
    info = {}
    info["commit"] = run_command("git rev-parse --short HEAD") or "unknown"
    info["branch"] = run_command("git rev-parse --abbrev-ref HEAD") or "unknown"
    status = run_command("git status --porcelain")
    info["dirty"] = bool(status) if status is not None else False
    info["message"] = run_command("git log -1 --pretty=%B") or ""
    info["message"] = info["message"].split("\n")[0][:80]
    return info


def get_gpu_info():
    """Get GPU inventory and CUDA version if available."""
    if not torch.cuda.is_available():
        return {"available": False}

    names = []
    memory_gb = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        names.append(props.name)
        memory_gb.append(props.total_memory / (1024**3))

    return {
        "available": True,
        "count": torch.cuda.device_count(),
        "names": names,
        "memory_gb": memory_gb,
        "cuda_version": torch.version.cuda or "unknown",
    }


def get_system_info():
    """Get basic host and runtime information."""
    return {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "user": os.environ.get("USER") or os.environ.get("USERNAME") or "unknown",
        "working_dir": os.getcwd(),
    }


def generate_header():
    """Generate markdown header for the report."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_info = get_git_info()
    gpu_info = get_gpu_info()
    sys_info = get_system_info()

    header = f"""# Training Report

Generated: {timestamp}

## Environment

### Git Information
- Branch: {git_info["branch"]}
- Commit: {git_info["commit"]} {"(dirty)" if git_info["dirty"] else "(clean)"}
- Message: {git_info["message"]}

### Hardware
- Platform: {sys_info["platform"]}
- CPUs: {sys_info["cpu_count"]} cores ({sys_info["cpu_count_logical"]} logical)
- Memory: {sys_info["memory_gb"]:.1f} GB
"""

    if gpu_info.get("available"):
        gpu_names = ", ".join(sorted(set(gpu_info["names"])))
        total_vram = sum(gpu_info["memory_gb"])
        header += f"""- GPUs: {gpu_info["count"]}x {gpu_names}
- GPU Memory: {total_vram:.1f} GB total
- CUDA Version: {gpu_info["cuda_version"]}
"""
    else:
        header += "- GPUs: None available\n"

    header += f"""
### Software
- Python: {sys_info["python_version"]}
- PyTorch: {sys_info["torch_version"]}

"""
    return header


def slugify(text):
    """Slugify a title into a file name-friendly token."""
    return text.lower().replace(" ", "-")


def extract_timestamp(content, prefix):
    """Extract a timestamp line from markdown content."""
    for line in content.split("\n"):
        if line.startswith(prefix):
            time_str = line.split(":", 1)[1].strip()
            try:
                return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
    return None


class Report:
    """Simple report logger and markdown assembler for experiment runs."""

    def __init__(self, report_dir):
        os.makedirs(report_dir, exist_ok=True)
        self.report_dir = report_dir

    def log(self, section, data):
        """Log one section to a markdown file."""
        file_name = f"{slugify(section)}.md"
        file_path = os.path.join(self.report_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"## {section}\n")
            f.write(
                f"timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )
            for item in data:
                if not item:
                    continue
                if isinstance(item, str):
                    f.write(item)
                    if not item.endswith("\n"):
                        f.write("\n")
                    continue

                for key, value in item.items():
                    if isinstance(value, float):
                        value_str = f"{value:.4f}"
                    elif isinstance(value, int) and value >= 10000:
                        value_str = f"{value:,.0f}"
                    else:
                        value_str = str(value)
                    f.write(f"- {key}: {value_str}\n")
            f.write("\n")
        return file_path

    def generate(self):
        """Assemble header + all section files into report.md."""
        report_file = os.path.join(self.report_dir, "report.md")
        header_file = os.path.join(self.report_dir, "header.md")

        preferred = set(SECTION_ORDER)
        present_sections = [
            name
            for name in os.listdir(self.report_dir)
            if name.endswith(".md") and name not in {"header.md", "report.md"}
        ]

        ordered_sections = [name for name in SECTION_ORDER if name in present_sections]
        ordered_sections.extend(
            sorted(name for name in present_sections if name not in preferred)
        )

        start_time = None
        end_time = None

        with open(report_file, "w", encoding="utf-8") as out_file:
            if os.path.exists(header_file):
                with open(header_file, "r", encoding="utf-8") as f:
                    header_content = f.read()
                    out_file.write(header_content)
                    start_time = extract_timestamp(header_content, "Run started:")
            else:
                out_file.write(generate_header())
                out_file.write("Run started: unknown\n\n---\n\n")

            for file_name in ordered_sections:
                section_path = os.path.join(self.report_dir, file_name)
                with open(section_path, "r", encoding="utf-8") as in_file:
                    section = in_file.read()
                out_file.write(section)
                if not section.endswith("\n"):
                    out_file.write("\n")
                out_file.write("\n")

                section_ts = extract_timestamp(section, "timestamp:")
                if section_ts is not None:
                    end_time = section_ts

            out_file.write("## Summary\n\n")
            out_file.write(f"- Sections included: {len(ordered_sections)}\n")
            if start_time and end_time:
                duration = end_time - start_time
                total_seconds = int(duration.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                out_file.write(f"- Total wall clock time: {hours}h{minutes}m\n")
            else:
                out_file.write("- Total wall clock time: unknown\n")

        shutil.copy(report_file, "report.md")
        print(f"Generated report: {report_file}")
        return report_file

    def reset(self):
        """Reset report directory content for a fresh run."""
        for file_name in os.listdir(self.report_dir):
            if not file_name.endswith(".md"):
                continue
            file_path = os.path.join(self.report_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

        header_file = os.path.join(self.report_dir, "header.md")
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(header_file, "w", encoding="utf-8") as f:
            f.write(generate_header())
            f.write(f"Run started: {start_time}\n\n---\n\n")

        print(f"Reset report directory and wrote header: {header_file}")


def get_report(report_dir=None):
    """Get a report instance for the current project."""
    if report_dir is None:
        report_dir = os.path.join("outputs", "report")
    return Report(report_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate or reset medical segmentation training reports."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="generate",
        choices=["generate", "reset"],
        help="Operation to perform (default: generate)",
    )
    parser.add_argument(
        "--report_dir",
        default=os.path.join("outputs", "report"),
        help="Directory where section markdown files and report.md are stored",
    )
    args = parser.parse_args()

    report = get_report(args.report_dir)
    if args.command == "generate":
        report.generate()
    else:
        report.reset()
