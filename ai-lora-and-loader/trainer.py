import logging
from pathlib import Path
import json
from typing import List, Dict
from transformers import (
	AutoTokenizer, 
	AutoModelForCausalLM, 
	TrainingArguments, 
	Trainer,
	BitsAndBytesConfig,
	DataCollatorForLanguageModeling,
	AutoConfig
)
from datasets import Dataset
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

class ModelFinetuner:
	def __init__(self, model_path: str,
				 output_dir: str,
				 save_mode: str = "lora",
				 quantization: str = "none",
				 training_config: dict = None,
				 lora_config: dict = None):
		self.model_path = model_path
		self.output_dir = output_dir
		self.save_mode = save_mode.lower()
		self.quantization = quantization.lower()
		self.training_config = training_config or {}
		self.lora_config = lora_config or {}
		
		if self.save_mode not in ["lora", "merge", "both"]:
			raise ValueError("save_mode must be one of: lora, merge, both")
		if self.quantization not in ["none", "4bit", "8bit"]:
			raise ValueError("quantization must be one of: none, 4bit, 8bit")
			
		self.setup_logging()
		
	def setup_logging(self):
		self.logger = logging.getLogger(__name__)
		
	def load_dataset(self, dataset_path: str) -> Dataset:
		"""加载数据集并转换为HuggingFace数据集格式"""
		with open(dataset_path, 'r', encoding='utf-8') as f:
			raw_data = json.load(f)
			
		# 转换为训练所需格式
		formatted_data = []
		for item in raw_data:
			conversation = item['conversations']
			# 对话格式
			formatted_text = "<|im_start|>system\n你是一个代码助手。<|im_end|>\n"
			for turn in conversation:
				formatted_text += f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n"
			formatted_data.append(formatted_text)
		
		# 创建数据集
		dataset = Dataset.from_list([{"text": text} for text in formatted_data])
		
		# 对数据集进行tokenize处理
		def tokenize_function(examples):
			return self.tokenizer(
				examples["text"],
				truncation=True,
				max_length=2048,
				padding=False,
				return_tensors=None,
			)
		
		# 应用tokenize处理
		tokenized_dataset = dataset.map(
			tokenize_function,
			remove_columns=["text"],
			desc="Tokenizing dataset",
		)
		
		return tokenized_dataset
		
	def prepare_model(self):
		"""准备模型和tokenizer"""
		# 加载tokenizer
		self.tokenizer = AutoTokenizer.from_pretrained(
			self.model_path,
			trust_remote_code=True,
			pad_token='<|extra_0|>'
		)
		
		# 准备模型加载参数
		model_kwargs = {
			"trust_remote_code": True,
			"torch_dtype": torch.float16,
			"device_map": "auto"
		}
		
		# 根据量化设置添加量化配置
		if self.quantization != "none":
			config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
			
			# 检查模型自带的量化配置
			if hasattr(config, 'quantization_config'):
				quantization_config = config.quantization_config
				self.logger.info(f"使用模型自带量化配置: {quantization_config}")
			else:
				# 根据参数设置量化配置
				if self.quantization == "4bit":
					quantization_config = BitsAndBytesConfig(
						load_in_4bit=True,
						bnb_4bit_compute_dtype=torch.float16,
						bnb_4bit_use_double_quant=True,
						bnb_4bit_quant_type="nf4"
					)
				else:  # 8bit
					quantization_config = BitsAndBytesConfig(
						load_in_8bit=True,
						llm_int8_threshold=6.0
					)
				self.logger.info(f"使用指定量化配置: {self.quantization}")
				
			model_kwargs["quantization_config"] = quantization_config
		
		# 加载模型
		self.model = AutoModelForCausalLM.from_pretrained(
			self.model_path,
			**model_kwargs
		)
		
		# 如果使用量化，先进行prepare
		if self.quantization != "none":
			self.model = prepare_model_for_kbit_training(self.model)
		
		# 从配置文件读取LoRA参数
		lora_config = LoraConfig(
			r=int(self.lora_config.get('r', 8)),
			lora_alpha=int(self.lora_config.get('lora_alpha', 32)),
			target_modules=self.lora_config.get('target_modules', '').split(','),
			lora_dropout=float(self.lora_config.get('lora_dropout', 0.05)),
			bias="none",
			task_type="CAUSAL_LM",
			inference_mode=False
		)
		
		# 应用LoRA配置
		self.model = get_peft_model(self.model, lora_config)
		
		# 设置训练参数
		self.model.config.use_cache = False  # 禁用KV缓存以启用梯度
		self.model.enable_input_require_grads()  # 启用输入梯度
		
		# 确保所有LoRA参数可训练
		for name, param in self.model.named_parameters():
			if "lora" in name or "adapter" in name:
				param.requires_grad = True
			else:
				param.requires_grad = False
		
		# 打印参数信息
		trainable_params = 0
		all_param = 0
		for name, param in self.model.named_parameters():
			all_param += param.numel()
			if param.requires_grad:
				trainable_params += param.numel()
				self.logger.info(f"可训练参数: {name}")
		self.logger.info(
			f"可训练参数总数: {trainable_params:,d} ({100 * trainable_params / all_param:.2f}%)"
		)
		
	def save_lora(self):
		"""保存LoRA权重"""
		self.logger.info(f"保存LoRA权重到 {self.output_dir}/lora")
		self.model.save_pretrained(f"{self.output_dir}/lora")
		self.tokenizer.save_pretrained(f"{self.output_dir}/lora")
	
	def save_merged(self):
		"""保存合并后的完整模型"""
		# 合并LoRA权重到基础模型
		self.logger.info("合并LoRA权重到基础模型...")
		merged_model = self.model.merge_and_unload()
		
		# 保存完整模型
		self.logger.info(f"保存完整模型到 {self.output_dir}/full")
		merged_model.save_pretrained(
			f"{self.output_dir}/full",
			safe_serialization=True,
			save_config=True,
		)
		self.tokenizer.save_pretrained(f"{self.output_dir}/full")
		
		# 复制原始模型的其他必要文件
		self.logger.info("复制模型配置文件...")
		import shutil
		src_path = Path(self.model_path)
		dst_path = Path(f"{self.output_dir}/full")
		
		# 复制必要的配置文件
		config_files = [
			"config.json",
			"generation_config.json",
			"tokenizer_config.json",
			"tokenizer.json",
			"special_tokens_map.json"
		]
		
		for file in config_files:
			if (src_path / file).exists():
				shutil.copy2(src_path / file, dst_path / file)
				self.logger.info(f"已复制 {file}")
	
	def train(self, dataset_path: str):
		"""训练模型"""
		# 先准备模型和tokenizer
		self.prepare_model()
		# 然后加载数据集
		dataset = self.load_dataset(dataset_path)
		
		# 从配置文件读取训练参数
		training_args = TrainingArguments(
			output_dir=self.output_dir,
			num_train_epochs=float(self.training_config.get('num_train_epochs', 3)),
			per_device_train_batch_size=int(self.training_config.get('per_device_train_batch_size', 2)),
			gradient_accumulation_steps=int(self.training_config.get('gradient_accumulation_steps', 8)),
			learning_rate=float(self.training_config.get('learning_rate', 1e-4)),
			weight_decay=float(self.training_config.get('weight_decay', 0.01)),
			warmup_ratio=float(self.training_config.get('warmup_ratio', 0.1)),
			logging_steps=int(self.training_config.get('logging_steps', 10)),
			save_steps=int(self.training_config.get('save_steps', 100)),
			save_total_limit=int(self.training_config.get('save_total_limit', 3)),
			fp16=True,
			remove_unused_columns=False,
			gradient_checkpointing=True,
			optim="paged_adamw_32bit",
			lr_scheduler_type="cosine",
			max_grad_norm=0.3,
			ddp_find_unused_parameters=False,
			torch_compile=False,
			report_to="none"
		)
		
		# 开始训练
		trainer = Trainer(
			model=self.model,
			args=training_args,
			train_dataset=dataset,
			data_collator=DataCollatorForLanguageModeling(
				tokenizer=self.tokenizer,
				mlm=False
			)
		)
		
		self.logger.info("开始训练模型...")
		trainer.train()
		
		# 根据save_mode保存模型
		if self.save_mode in ["lora", "both"]:
			self.save_lora()
			
		if self.save_mode in ["merge", "both"]:
			self.save_merged() 