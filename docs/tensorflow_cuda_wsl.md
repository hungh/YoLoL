### Instructions how to install tensorflow with CUDA on WSL
1. Ensure nvdia drivers are installed and working. Check with `nvidia-smi`

```bash
nvidia-smi
Wed Feb  4 19:34:13 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.64.04              Driver Version: 577.00         CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 Ti     On  |   00000000:02:00.0  On |                  N/A |
|  0%   31C    P5              4W /  180W |    1689MiB /  16311MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```
2. Install CUDA toolkit
For the above 12.9 driver, use CUDA 12.9, link [here](https://developer.nvidia.com/cuda-12-9-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network)
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9
```
3. Install tensorflow with CUDA support
```bash
pip install tensorflow[and-cuda]
```

4. Verify installation
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
Look for GPU in the device list.
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```
5. (Optional) Increase WSL Memory and CPU count 
Create or edit `C:\Users\<username>\.wslconfig`:
```
[wsl2]
memory=24GB
processors=20
```
Then restart WSL:
```bash
wsl --shutdown
```

