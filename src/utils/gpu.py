"""GPU utilities."""
import torch


def get_device() -> str:
    """Return 'cuda' if GPU available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def print_gpu_info():
    """Print GPU memory info."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {name}")
            print(f"  Memory: {mem_allocated:.2f} GB allocated / {mem_reserved:.2f} GB reserved / {mem_total:.2f} GB total")


def clear_cache():
    """Clear GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
