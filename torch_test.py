import torch

try:
    print(f"PyTorch version: {torch.__version__}")
    print("-" * 30)
    
    is_available = torch.cuda.is_available()
    print(f"CUDA available: {is_available}")

    if is_available:
        print(f"CUDA version by PyTorch: {torch.version.cuda}")
        print(f"cuDNN version by PyTorch: {torch.backends.cudnn.version()}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Please check drivers and installation.")

except Exception as e:
    print(f"An error occurred: {e}")