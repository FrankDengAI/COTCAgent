# COTCAgent Web界面

## 概述

这是一个完整的Web界面，用于与COTCAgent医疗时序数据分析系统进行交互。界面提供了直观的用户体验，包括实时聊天、分析结果展示和医疗建议。

## 功能特性

### 🎯 核心功能
- **实时对话**: 与AI助手进行自然语言交互
- **数据分析**: 自动分析患者时序健康数据
- **风险评估**: 显示疾病风险评分和置信度
- **医疗建议**: 提供专业的医疗建议和问诊问题
- **响应式设计**: 适配各种设备屏幕

### 🎨 界面特色
- **现代化UI**: 渐变背景、圆角设计、动画效果
- **实时状态**: 显示系统在线状态和处理进度
- **智能提示**: 自动生成问诊问题
- **结果可视化**: 清晰展示分析结果和风险等级

## 文件结构

```
COTCAgent/
├── web_interface.html      # 前端界面
├── backend_api.py         # Flask后端API
├── start_server.py        # 服务器启动脚本
├── requirements.txt       # Python依赖
├── cotc_agent.py         # COTCAgent核心
├── example_usage.py       # 使用示例
└── README_WebInterface.md # 本文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务器

```bash
python start_server.py
```

或者手动启动：

```bash
python backend_api.py
```

### 3. 访问界面

打开浏览器访问: http://localhost:5000

## API接口

### 获取患者信息
```
GET /api/patient/info
```

### 分析用户查询
```
POST /api/analysis/query
Content-Type: application/json

{
    "query": "我最近肠胃老是疼，而且头也经常晕"
}
```

### 获取分析状态
```
GET /api/analysis/status
```

### 健康检查
```
GET /api/health
```

## 使用说明

### 1. 患者信息面板
- 显示患者ID、总指标数、现有诊断
- 实时显示系统状态

### 2. 对话界面
- 在输入框中描述症状
- 按Enter键或点击发送按钮
- AI会分析并返回结果

### 3. 分析结果
- **风险评估**: 显示疾病风险分数和置信度
- **医疗建议**: 提供专业的医疗建议
- **问诊问题**: 点击问题可进行后续对话

### 4. 交互流程
1. 用户输入症状描述
2. AI分析时序数据
3. 显示风险评估结果
4. 提供医疗建议
5. 生成问诊问题
6. 支持后续对话

## 技术架构

### 前端技术
- **HTML5**: 语义化结构
- **CSS3**: 现代样式和动画
- **JavaScript**: 交互逻辑和API调用
- **响应式设计**: 适配移动端

### 后端技术
- **Flask**: Python Web框架
- **Flask-CORS**: 跨域请求支持
- **异步处理**: 支持长时间分析任务
- **RESTful API**: 标准化接口设计

### 核心集成
- **COTCAgent**: 医疗分析核心
- **DeepSeek API**: AI代码生成
- **患者数据**: JSON格式存储

## 自定义配置

### 修改API配置
编辑 `backend_api.py` 中的配置：

```python
config = DeepSeekConfig(
    api_key='your_api_key',
    api_base="https://api.deepseek.com/v1/chat/completions",
    model="deepseek-chat",
    max_tokens=2000,
    temperature=0.7,
    timeout=180,
    save_temp_files=True
)
```

### 修改界面样式
编辑 `web_interface.html` 中的CSS样式：

```css
/* 修改主题色 */
:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
}
```

### 添加新功能
1. 在 `backend_api.py` 中添加新的API端点
2. 在 `web_interface.html` 中添加前端逻辑
3. 更新JavaScript处理新的API响应

## 故障排除

### 常见问题

1. **依赖安装失败**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **端口被占用**
   ```bash
   # 修改端口
   app.run(debug=True, host='0.0.0.0', port=5001)
   ```

3. **API连接失败**
   - 检查DeepSeek API密钥
   - 确认网络连接
   - 查看控制台错误信息

4. **患者数据加载失败**
   - 确认 `patient_data/patient_0001.json` 存在
   - 检查JSON格式是否正确

### 调试模式

启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 部署建议

### 生产环境
1. 使用Gunicorn或uWSGI
2. 配置Nginx反向代理
3. 设置环境变量
4. 启用HTTPS

### 示例部署命令
```bash
# 使用Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 backend_api:app

# 使用Docker
docker build -t cotcagent-web .
docker run -p 5000:5000 cotcagent-web
```

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

MIT License

## 联系方式

如有问题，请提交Issue或联系开发团队。
