"""
COTCAgent Web服务器启动脚本
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """检查依赖是否安装"""
    try:
        import flask
        import flask_cors
        print("[SUCCESS] Flask依赖已安装")
        return True
    except ImportError:
        print("[ERROR] Flask依赖未安装")
        return False

def install_dependencies():
    """安装依赖"""
    print("正在安装依赖...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("[SUCCESS] 依赖安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 依赖安装失败: {e}")
        return False

def setup_templates():
    """设置模板目录"""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # 移动HTML文件到templates目录
    if Path("web_interface.html").exists():
        import shutil
        shutil.move("web_interface.html", "templates/web_interface.html")
        print("[SUCCESS] HTML模板已移动到templates目录")

def start_server():
    """启动服务器"""
    print("\n" + "="*50)
    print("COTCAgent Web服务器启动")
    print("="*50)
    
    # 检查依赖
    if not check_dependencies():
        print("Flask依赖未安装，请手动运行: pip install flask flask-cors")
        return False
    
    # 设置模板
    setup_templates()
    
    # 启动服务器
    print("\n启动Flask服务器...")
    print("服务器地址: http://localhost:5000")
    print("按 Ctrl+C 停止服务器")
    print("\n正在打开浏览器...")
    
    # 延迟打开浏览器
    def open_browser():
        time.sleep(2)
        webbrowser.open("http://localhost:5000")
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # 启动Flask应用
    try:
        from backend_api import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"服务器启动失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("COTCAgent Web界面启动器")
    print("="*30)
    
    # 检查必要文件
    required_files = [
        "cotc_agent.py",
        "backend_api.py", 
        "templates/web_interface.html",
        "patient_data/patient_0001.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("[ERROR] 缺少必要文件:")
        for file in missing_files:
            print(f"  - {file}")
        print("\n请确保所有必要文件都存在")
        sys.exit(1)
    
    print("[SUCCESS] 所有必要文件检查通过")
    
    # 启动服务器
    start_server()
