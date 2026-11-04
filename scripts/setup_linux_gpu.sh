#!/usr/bin/env bash
# Setup on a Linux NVIDIA machine (Kaggle, Colab) where PyTorch with CUDA is preinstalled.
# Installs the project's Python packages (keeping the platform's torch) and the Vulkan
# loader/ICD files SAPIEN needs to render on NVIDIA GPUs, following ManiSkill's Colab/Docker setup.
# Run as root (the default on Kaggle/Colab).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if ! ldconfig -p | grep -q "libvulkan.so.1"; then
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends libvulkan1 >/dev/null
fi

mkdir -p /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d /etc/vulkan/implicit_layer.d
[ -f /usr/share/vulkan/icd.d/nvidia_icd.json ] || cat > /usr/share/vulkan/icd.d/nvidia_icd.json <<'EOF'
{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.2.155"}}
EOF
[ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ] || cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'EOF'
{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}
EOF
[ -f /etc/vulkan/implicit_layer.d/nvidia_layers.json ] || cat > /etc/vulkan/implicit_layer.d/nvidia_layers.json <<'EOF'
{"file_format_version": "1.0.0", "layer": {"name": "VK_LAYER_NV_optimus", "type": "INSTANCE",
 "library_path": "libGLX_nvidia.so.0", "api_version": "1.2.155", "implementation_version": "1",
 "description": "NVIDIA Optimus layer",
 "functions": {"vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr", "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"},
 "enable_environment": {"__NV_PRIME_RENDER_OFFLOAD": "1"}, "disable_environment": {"DISABLE_LAYER_NV_OPTIMUS_1": ""}}}
EOF

# Same versions as requirements.txt, minus torch/torchvision (use the platform's CUDA build)
# and pin (SAPIEN's Linux wheel bundles Pinocchio).
grep -v -E "^(torch|torchvision|pin)==" "$ROOT/requirements.txt" | grep -v "^#" | grep -v "^$" > /tmp/requirements-gpu.txt
pip install -q -r /tmp/requirements-gpu.txt

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
