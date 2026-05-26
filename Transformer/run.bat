@echo off
chcp 65001 >nul
echo ============================================
echo   Transformer 学习程序快速启动
echo ============================================
echo.
echo 正在激活 conda 环境并运行程序...
echo.

call conda activate transformer_learn
if errorlevel 1 (
    echo [错误] 无法激活 transformer_learn 环境
    echo 请先运行：conda env create -f environment.yml
    pause
    exit /b 1
)

echo [成功] 环境激活完成
echo.
echo 请选择要运行的程序:
echo.
echo   1. main.py              - 工作流程演示 (推荐)
echo   2. train_transformer.py - 训练演示
echo   3. visualize_attention.py - 注意力可视化
echo   4. 打开 Jupyter Notebook
echo   5. 退出
echo.
set /p choice="请输入选项 (1-5): "

if "%choice%"=="1" (
    echo.
    echo 运行 main.py...
    python main.py
) else if "%choice%"=="2" (
    echo.
    echo 运行 train_transformer.py...
    python train_transformer.py
) else if "%choice%"=="3" (
    echo.
    echo 运行 visualize_attention.py...
    python visualize_attention.py
) else if "%choice%"=="4" (
    echo.
    echo 启动 Jupyter Notebook...
    jupyter notebook transformer_explained.ipynb
) else if "%choice%"=="5" (
    exit /b 0
) else (
    echo 无效选项
)

echo.
pause
