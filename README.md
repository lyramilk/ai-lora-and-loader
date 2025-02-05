# 微调助手

代码助手微调工具，支持代码收集、模型微调和在线服务。

## 项目文件

	weitiao/
	├── config.ini          # 配置文件
	├── main.py            # 主程序入口
	├── collector.py       # 代码收集模块
	├── trainer.py         # 模型训练模块
	├── service.py         # 服务模块
	└── static/            # 静态文件目录
		└── index.html     # Web界面

## 功能特性

	- 代码收集
		- 支持扫描Git仓库
		- 自动识别代码文件
		- 生成训练数据集

	- 模型训练
		- 支持LoRA微调
		- 支持模型量化
		- 可配置训练参数
		- 支持多种保存模式

	- 在线服务
		- Web界面交互
		- 流式输出
		- 支持停止生成
		- 兼容OpenAI API

## 快速开始

	1. 安装依赖
		```bash
		pip install -r requirements.txt
		```

	2. 配置参数
		编辑config.ini文件，设置相关路径和参数：
		```ini
		[paths]
		data_dir = /path/to/git/repos      # Git仓库目录
		model_path = /path/to/base/model   # 基础模型路径
		output_dir = /path/to/output       # 输出目录
		```

	3. 收集代码
		```bash
		python main.py --collect_only
		```

	4. 训练模型
		```bash
		python main.py --train_only
		```

	5. 启动服务
		```bash
		python service.py
		```

## 配置说明(config.ini)
配置文件中所有配置都是默认配置，可以通过命令行参数覆盖。

### 路径配置 [paths]
	- data_dir: Git代码仓库的根目录
	- model_path: 基础模型路径
	- output_dir: 模型输出目录
	- dataset_path: 数据集文件路径

### 模型配置 [model]
	- save_mode: 模型保存模式(lora/merge/both)
	- quantization: 量化方式(none/4bit/8bit)

### 服务配置 [service]
	- port: 服务端口
	- host: 服务地址

### 训练配置 [training]
	- num_train_epochs: 训练轮数
	- per_device_train_batch_size: 批次大小
	- gradient_accumulation_steps: 梯度累积步数
	- learning_rate: 学习率
	- weight_decay: 权重衰减
	- warmup_ratio: 预热比例
	- logging_steps: 日志记录步数
	- save_steps: 保存步数
	- save_total_limit: 检查点数量限制

### LoRA配置 [lora]
	- r: LoRA秩
	- lora_alpha: 缩放因子
	- lora_dropout: Dropout率
	- target_modules: 目标模块列表

## 使用示例

### 完整流程
	```bash
	# 收集代码并训练
	python main.py --data_dir /path/to/repos --model_path /path/to/model

	# 仅收集代码
	python main.py --collect_only --data_dir /path/to/repos

	# 仅训练模型
	python main.py --train_only --model_path /path/to/model

	# 启动服务
	python service.py --model /path/to/model --lora /path/to/lora
	```

### Web界面
	访问 http://localhost:8080 使用Web界面：
	- 支持调整生成参数
	- 实时显示生成结果
	- 可随时停止生成
	- 支持代码高亮显示

## 注意事项

1. 内存需求
	- 3B模型建议32GB以上内存
	- 使用量化可降低内存需求
	- 可通过batch_size调整

2. 显存需求
	- 3B模型建议24GB以上显存
	- 4bit量化可降至12GB
	- 8bit量化可降至16GB

3. 训练建议
	- 建议使用LoRA训练
	- 根据显存调整batch_size
	- 适当增加累积步数
	- 注意调整学习率

4. 服务部署
	- 建议使用量化模型
	- 确保端口可访问
	- 考虑添加认证
	- 注意日志监控

## 常见问题

	Q: 显存不足怎么办？
		A: 可以尝试：
		1. 使用量化(4bit/8bit)
		2. 减小batch_size
		3. 增加gradient_accumulation_steps

	Q: 训练效果不好怎么办？
		A: 可以尝试：
		1. 增加训练轮数
		2. 调整学习率
		3. 增加LoRA秩(r)
		4. 改进训练数据质量

	Q: 生成速度慢怎么办？
		A: 可以尝试：
		1. 使用更小的模型
		2. 启用量化推理
		3. 调整max_tokens
		4. 使用更好的硬件
