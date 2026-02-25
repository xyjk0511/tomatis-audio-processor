# FFmpeg 自动安装脚本
# 此脚本会下载 FFmpeg 并配置到系统 PATH

Write-Host "=== FFmpeg 安装脚本 ===" -ForegroundColor Cyan
Write-Host ""

# 设置路径
$ffmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
$downloadPath = "$env:TEMP\ffmpeg-release-essentials.zip"
$extractPath = "C:\ffmpeg"
$ffmpegBinPath = "$extractPath\bin"

# 检查是否已安装
$ffmpegExe = "$ffmpegBinPath\ffmpeg.exe"
if (Test-Path $ffmpegExe) {
    Write-Host "✓ FFmpeg 已安装在: $extractPath" -ForegroundColor Green
} else {
    Write-Host "正在下载 FFmpeg（约 100 MB，可能需要几分钟）..." -ForegroundColor Yellow
    
    try {
        # 下载 FFmpeg
        Invoke-WebRequest -Uri $ffmpegUrl -OutFile $downloadPath -UseBasicParsing
        Write-Host "✓ 下载完成" -ForegroundColor Green
        
        Write-Host "正在解压..." -ForegroundColor Yellow
        
        # 创建目标目录
        if (-not (Test-Path $extractPath)) {
            New-Item -ItemType Directory -Path $extractPath -Force | Out-Null
        }
        
        # 解压
        Expand-Archive -Path $downloadPath -DestinationPath "$env:TEMP\ffmpeg_temp" -Force
        
        # 查找解压后的文件夹（通常是 ffmpeg-x.x.x-essentials_build）
        $extractedFolder = Get-ChildItem "$env:TEMP\ffmpeg_temp" -Directory | Select-Object -First 1
        
        # 移动文件到目标位置
        Copy-Item -Path "$($extractedFolder.FullName)\*" -Destination $extractPath -Recurse -Force
        
        Write-Host "✓ 解压完成" -ForegroundColor Green
        
        # 清理临时文件
        Remove-Item $downloadPath -ErrorAction SilentlyContinue
        Remove-Item "$env:TEMP\ffmpeg_temp" -Recurse -ErrorAction SilentlyContinue
        
    } catch {
        Write-Host "✗ 安装失败: $_" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "=== 配置环境变量 ===" -ForegroundColor Cyan

# 检查 PATH 中是否已包含 FFmpeg
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")

if ($currentPath -notlike "*$ffmpegBinPath*") {
    Write-Host "正在添加 FFmpeg 到用户 PATH..." -ForegroundColor Yellow
    
    try {
        # 添加到用户环境变量
        $newPath = "$currentPath;$ffmpegBinPath"
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        
        # 更新当前会话的 PATH
        $env:Path = "$env:Path;$ffmpegBinPath"
        
        Write-Host "✓ PATH 配置成功" -ForegroundColor Green
    } catch {
        Write-Host "✗ PATH 配置失败: $_" -ForegroundColor Red
        Write-Host "请手动添加到 PATH: $ffmpegBinPath" -ForegroundColor Yellow
    }
} else {
    Write-Host "✓ FFmpeg 已在 PATH 中" -ForegroundColor Green
    $env:Path = "$env:Path;$ffmpegBinPath"
}

Write-Host ""
Write-Host "=== 验证安装 ===" -ForegroundColor Cyan

# 验证 FFmpeg
$ffmpegVersion = & $ffmpegExe -version 2>&1 | Select-Object -First 1

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ FFmpeg 安装成功" -ForegroundColor Green
    Write-Host "版本: $ffmpegVersion" -ForegroundColor Cyan
} else {
    Write-Host "✗ FFmpeg 验证失败" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== 测试 FFmpeg ===" -ForegroundColor Cyan
Write-Host "示例命令（转换音频格式）:" -ForegroundColor Yellow
Write-Host "ffmpeg -i input.flac -ar 48000 -ac 1 -c:a pcm_s16le output.wav"
Write-Host ""

Write-Host "=== 安装完成 ===" -ForegroundColor Green
Write-Host ""
Write-Host "注意: 如果在新终端中 ffmpeg 命令仍未找到，请重启终端。" -ForegroundColor Yellow
Write-Host ""
