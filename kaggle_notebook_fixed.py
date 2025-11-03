# FrugalNeRF Kaggle Notebook

## Cell 1: Setup môi trường & clone repo

```python
# Kiểm tra GPU
!nvidia-smi

# Clone repo từ GitHub
!git clone https://github.com/haihoan2874/FrugalNeRF.git

# Chuyển vào thư mục FrugalNeRF/FrugalNeRF (cấu trúc lồng nhau sau clone)
import os
os.chdir('FrugalNeRF/FrugalNeRF')
print("Đã chuyển vào thư mục FrugalNeRF/FrugalNeRF")
print("Thư mục hiện tại:", os.getcwd())
```

## Cell 2: Cài dependencies

```python
# Cài đặt PyTorch (phiên bản phù hợp với CUDA)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Force reinstall torchvision cu118 để match với PyTorch cu118
!pip uninstall torchvision -y
!pip install torchvision --index-url https://download.pytorch.org/whl/cu118

# Cài đặt các thư viện khác
!pip install tqdm opencv-python scikit-image lpips configargparse imageio matplotlib plyfile

# Force install NumPy 1.26.4 để tương thích với các module cũ
!pip install 'numpy==1.26.4' --force-reinstall

print("Đã cài đặt xong tất cả dependencies!")
```

## Cell 3: Train FrugalNeRF

```python
# Tạo thư mục logs nếu chưa có
!mkdir -p logs

# Chạy lệnh train FrugalNeRF với upsamp_list fix
!python train.py --config configs/your_own_data.txt --datadir /kaggle/input/frugalnerf-habitat67/frugalnerf_data --basedir ./logs --n_iters 3000 --progress_refresh_rate 10 --upsamp_list 2000 --upsamp_list 3000

print("Đã hoàn thành quá trình train FrugalNeRF!")
```

## Cell 4: Render & đánh giá

```python
# Chạy render test images
!python train.py --config configs/your_own_data.txt --datadir /kaggle/input/frugalnerf-habitat67/frugalnerf_data --basedir ./logs --render_test 1 --ckpt ./logs/tensorf_xxx_VM/tensorf_xxx_VM.th

# Đọc và in ra metrics từ mean.txt
import os
mean_file = './logs/tensorf_xxx_VM/mean.txt'
if os.path.exists(mean_file):
    with open(mean_file, 'r') as f:
        content = f.read()
        print("Metrics từ mean.txt:")
        print(content)
        # Tìm và in PSNR, SSIM, LPIPS nếu có
        lines = content.split('\n')
        for line in lines:
            if 'PSNR' in line or 'SSIM' in line or 'LPIPS' in line:
                print(line)
else:
    print("Không tìm thấy file mean.txt. Kiểm tra lại đường dẫn.")

print("Đã hoàn thành render và đánh giá!")
```
