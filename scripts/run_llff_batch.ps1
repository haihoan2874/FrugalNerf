# Set environment variables for all Python processes
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "1"

# Set Python paths with forward slashes for cross-platform compatibility 
$pythonPath = "python"
$configPath = "configs"

<#
Run LLFF experiments (train -> render -> compute metrics) for multiple scenes and view counts.

Usage examples:
# Run train+render+metrics for default scenes (horn,fern,leaves,room) and views 2,3,4
.\run_llff_batch.ps1

# Only render + compute metrics (skip training) if checkpoints already exist
.\run_llff_batch.ps1 -SkipTrain

# Custom scenes and views
.\run_llff_batch.ps1 -Scenes @("fern","horn") -Views @(2,3)
#>

param(
    [string[]]$Scenes,
    [int[]]$Views,
    [int]$n_iters,
    [string]$OutCsv,
    [switch]$SkipTrain
)

# Set default values if not provided
if (-not $Scenes) { $Scenes = @("horn", "fern", "leaves", "room") }
if (-not $Views) { $Views = @(2, 3, 4) }
if (-not $n_iters) { $n_iters = 2000 }
if (-not $OutCsv) { $OutCsv = "frugal_results.csv" }

# Set FrugalNeRF optimal parameters
$chunk_size = 8192      # Larger batch size for better convergence
$N_samples = 64         # Number of sample points per ray
$N_importance = 64      # Number of importance samples for better details
$downsample = 8.0      # Higher downsample ratio for faster training
$ndc_ray = 1           # Enable NDC ray parameterization
$n_iters_half = 1000   # Learning rate decay after these many iterations

# Novel view synthesis parameters
$novel_view_interval = 8  # Interval between training views for novel view testing
$spiral_radius = 0.8     # Radius of spiral path for novel views
$num_spiral_frames = 120  # Number of frames in spiral path for smoother video
$perturb_views = 1       # Enable view perturbation during training
$random_bg = 1           # Random background color for better generalization

$basedir = Get-Location
$datadir = Join-Path $basedir 'data\nerf_llff_data'

if (-not (Test-Path $OutCsv)) {
    "scene,views,expname,psnr,ssim,lpips,train_time_seconds,ckpt" | Out-File -FilePath $OutCsv -Encoding utf8
}

foreach ($scene in $Scenes) {
    $scenePath = Join-Path $datadir $scene
    # convert to forward-slash path for Python scripts that split on '/'
    $scenePathUnix = $scenePath -replace '\\', '/'
    if (-not (Test-Path $scenePath)) { Write-Warning "Scene not found: $scenePath - skipping"; continue }

    # collect images file list (use images_4 / images_8 if present, fall back to images)
    $imageDirs = @('images_8', 'images_4', 'images')
    $imgFolder = $null
    foreach ($d in $imageDirs) {
        $p = Join-Path $scenePath $d
        if (Test-Path $p) { $imgFolder = $p; break }
    }
    if ($imgFolder -ne $null) { $imgFolderUnix = $imgFolder -replace '\\', '/' }
    if ($imgFolder -eq $null) { Write-Warning "No images folder under $scenePath - skipping"; continue }

    $imgFiles = Get-ChildItem -Path $imgFolder -File | Sort-Object Name
    $imgCount = $imgFiles.Count
    if ($imgCount -lt 4) { Write-Warning "Scene $scene has only $imgCount images - results may be unreliable" }

    foreach ($v in $Views) {
        # Always use light config for faster training
        $exp = "${scene}_${v}v_light"
        $cfg = "configs\llff_light_${v}v.txt"

        # pick train frames evenly across available images for sparse view setting
        if ($v -le 1) { Write-Warning "FrugalNeRF requires at least 2 views"; continue }
        if ($v -gt 4) { Write-Warning "FrugalNeRF is designed for 2-4 views, performance may degrade"; }
        
        # Evenly space the views for better coverage
        $train_frames = @()
        for ($i = 0; $i -lt $v; $i++) {
            $idx = [math]::Floor(($i * ($imgCount - 1)) / ($v - 1))
            $train_frames += $idx
        }
        $train_args = $train_frames
        $test_frame = [math]::Floor($imgCount / 2)

        Write-Host "=== Scene=$scene | Views=$v | Exp=$exp ==="

        $runLog = "run_${scene}_${v}v.txt"
        $renderLog = "render_${scene}_${v}v.txt"

        if (-not $SkipTrain) {
            $start = Get-Date
            Write-Host "Training $scene with $v views (FrugalNeRF settings) -> exp=$exp"
            $train_cmd = "$pythonPath train.py --config $cfg --datadir $scenePathUnix --expname $exp --train_frame_num $($train_args -join ' ') --test_frame_num $test_frame --n_iters $n_iters --perturb $perturb_views"
            Write-Host $train_cmd
            & $pythonPath train.py --config $cfg --datadir $scenePathUnix --expname $exp --train_frame_num @($train_args) --test_frame_num $test_frame --n_iters $n_iters --perturb $perturb_views *>&1 | Tee-Object -FilePath $runLog
            $exit = $LASTEXITCODE
            $duration = (Get-Date) - $start
            if ($exit -ne 0) { Write-Warning "Training failed for $scene $v-view. See $runLog"; continue }
        }
        else {
            Write-Host "Skipping training for $scene $v-view (SkipTrain)"
            $duration = New-TimeSpan -Seconds 0
        }

        # find checkpoint
        $ckpt = Join-Path (Join-Path $basedir 'log') (Join-Path $exp "${exp}.th")
        if (-not (Test-Path $ckpt)) {
            $th = Get-ChildItem -Path (Join-Path $basedir 'log' -ChildPath $exp) -Filter '*.th' -File -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($th -eq $null) { Write-Warning "No checkpoint found for $exp - skipping"; continue } else { $ckpt = $th.FullName }
        }


        # --- Export mesh ---
        Write-Host "Exporting mesh for $exp"
        $mesh_cmd = "$pythonPath train.py --export_mesh 1 --ckpt `"$ckpt`" --config $cfg --datadir $scenePathUnix --expname $exp --dataset_name llff"
        Write-Host $mesh_cmd
        & $pythonPath train.py --export_mesh 1 --ckpt $ckpt --config $cfg --datadir $scenePathUnix --expname $exp --dataset_name llff *>&1 | Tee-Object -FilePath "mesh_${scene}_${v}v.txt"

        # --- Render test images ---
        Write-Host "Rendering test images for $exp"
        $render_cmd = "$pythonPath train.py --config $cfg --datadir $scenePathUnix --ckpt `"$ckpt`" --render_test 1 --expname $exp"
        Write-Host $render_cmd
        & $pythonPath train.py --config $cfg --datadir $scenePathUnix --ckpt $ckpt --render_test 1 --expname $exp *>&1 | Tee-Object -FilePath $renderLog

        $render_dir = Join-Path (Join-Path $basedir 'log') (Join-Path $exp 'imgs_test_all')
        if (-not (Test-Path $render_dir)) { Write-Warning "No render dir: $render_dir"; continue }


        # --- Create spiral video with improved novel view synthesis ---
        Write-Host "Rendering novel views for $exp with FrugalNeRF settings"
        
        # Generate spiral path with more frames and controlled radius 
        Write-Host "Rendering spiral path video..."
        & $pythonPath train.py --config $cfg --datadir $scenePathUnix --ckpt $ckpt --expname $exp --render_only 1 --render_spiral 1 --render_test 1 --N_samples $N_samples --N_importance $N_importance --chunk $chunk_size *>&1 | Tee-Object -FilePath "spiral_${scene}_${v}v.txt"

        # --- Compute comprehensive metrics including novel view evaluation ---
        Write-Host "Computing metrics for $exp (including novel view quality)"
        
        # Compute metrics for test views
        Write-Host "Evaluating test view reconstruction"
        $render_dirUnix = $render_dir -replace '\\', '/'
        & $pythonPath extra/compute_metrics.py --render_dir $render_dirUnix --gt_dir $imgFolderUnix --eval_type test *>&1 | Tee-Object -FilePath "metrics_test_${scene}_${v}v.txt"
        
        # Compute metrics for interpolated views (if ground truth available)
        $interp_dir = Join-Path (Join-Path $basedir 'log') (Join-Path $exp 'imgs_interp')
        if (Test-Path $interp_dir) {
            Write-Host "Evaluating novel view quality"
            $interp_dirUnix = $interp_dir -replace '\\', '/'
            & $pythonPath extra/compute_metrics.py --render_dir $interp_dirUnix --gt_dir $imgFolderUnix --eval_type novel *>&1 | Tee-Object -FilePath "metrics_novel_${scene}_${v}v.txt"
        }
        
        $meanfile = Join-Path $render_dir 'mean.txt'
        if (-not (Test-Path $meanfile)) { Write-Warning "mean.txt missing for $exp"; continue }
        $lines = Get-Content $meanfile
        $psnr = $lines[0]; $ssim = $lines[1]; $lpips = $lines[2]

        $seconds = [math]::Round($duration.TotalSeconds, 2)
        "$scene,$v,$exp,$psnr,$ssim,$lpips,$seconds,$ckpt" | Out-File -FilePath $OutCsv -Append -Encoding utf8

        Write-Host "Completed $scene $v-view: psnr=$psnr ssim=$ssim lpips=$lpips time=${seconds}s"
    }
}

Write-Host "Batch done. Results appended to $OutCsv" 
