sudo apt install python3.12 python3.12-dev python3.12-venv python3.12-full
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126

sudo apt-get --purge remove "*cublas*" "cuda*" -y 2>/dev/null

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-wsl-ubuntu-12-6-local_12.6.3-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-6-local_12.6.3-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update

sudo apt-get install -y cuda-toolkit-12-6

sudo apt install -y gcc-13 g++-13
CC=gcc-13 CXX=g++-13


CUDA_MATH_H=/usr/local/cuda/include/crt/math_functions.h

# see the current (unpatched) declarations first
grep -nE "sinpi\(double x\)|sinpif\(float x\)|cospi\(double x\)|cospif\(float x\)|rsqrt\(double x\)|rsqrtf\(float x\)" "$CUDA_MATH_H"

# back it up before touching it
sudo cp "$CUDA_MATH_H" "$CUDA_MATH_H.bak"

# add the matching noexcept(true) to all six declarations
sudo sed -i \
  -e 's/\bsinpi(double x);/sinpi(double x) noexcept (true);/' \
  -e 's/\bsinpif(float x);/sinpif(float x) noexcept (true);/' \
  -e 's/\bcospi(double x);/cospi(double x) noexcept (true);/' \
  -e 's/\bcospif(float x);/cospif(float x) noexcept (true);/' \
  -e 's/\brsqrt(double x);/rsqrt(double x) noexcept (true);/' \
  -e 's/\brsqrtf(float x);/rsqrtf(float x) noexcept (true);/' \
  "$CUDA_MATH_H"

# confirm all six now show noexcept
grep -nE "sinpi\(double x\) noexcept|sinpif\(float x\) noexcept|cospi\(double x\) noexcept|cospif\(float x\) noexcept|rsqrt\(double x\) noexcept|rsqrtf\(float x\) noexcept" "$CUDA_MATH_H"


export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export CUDAHOSTCXX=/usr/bin/g++-13
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/gcc-13"

MAMBA_FORCE_BUILD=TRUE MAMBA_KEEP_CUDA_BUILD=TRUE pip install mamba-ssm --no-build-isolation
