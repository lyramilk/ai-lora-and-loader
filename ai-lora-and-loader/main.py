import argparse
import logging
import configparser
from pathlib import Path
from collector import GitCodeCollector
from trainer import ModelFinetuner

def setup_logging():
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(levelname)s - %(message)s'
	)

def load_config(config_file: str = 'config.ini') -> configparser.ConfigParser:
	"""加载配置文件"""
	config = configparser.ConfigParser()
	config_path = Path(__file__).parent / config_file
	
	if not config_path.exists():
		raise FileNotFoundError(f"配置文件不存在: {config_path}")
		
	config.read(config_path, encoding='utf-8')
	return config

def main():
	# 加载配置
	config = load_config()
	
	# 设置命令行参数，允许覆盖配置文件
	parser = argparse.ArgumentParser(description='收集代码并微调模型')
	parser.add_argument('--data_dir', type=str,
						default=config.get('paths', 'data_dir'),
						help='Git仓库所在目录')
	parser.add_argument('--model_path', type=str,
						default=config.get('paths', 'model_path'),
						help='模型路径')
	parser.add_argument('--output_dir', type=str,
						default=config.get('paths', 'output_dir'),
						help='输出目录')
	parser.add_argument('--dataset_path', type=str,
						default=config.get('paths', 'dataset_path'),
						help='数据集保存路径')
	parser.add_argument('--collect_only', action='store_true',
						help='仅收集数据，不进行训练')
	parser.add_argument('--train_only', action='store_true',
						help='仅进行训练，不收集数据')
	parser.add_argument('--save_mode', type=str,
						default=config.get('model', 'save_mode'),
						choices=['lora', 'merge', 'both'],
						help='模型保存模式')
	parser.add_argument('--quantization', type=str,
						default=config.get('model', 'quantization'),
						choices=['none', '4bit', '8bit'],
						help='量化方式')
	
	args = parser.parse_args()
	setup_logging()
	logger = logging.getLogger(__name__)
	
	if not args.train_only:
		logger.info("开始收集代码...")
		collector = GitCodeCollector(args.data_dir)
		collector.run(args.dataset_path)
	
	if not args.collect_only:
		logger.info("开始训练模型...")
		# 从配置文件读取训练参数
		training_config = dict(config['training'])
		lora_config = dict(config['lora'])
		
		finetuner = ModelFinetuner(
			model_path=args.model_path,
			output_dir=args.output_dir,
			save_mode=args.save_mode,
			quantization=args.quantization,
			training_config=training_config,
			lora_config=lora_config
		)
		finetuner.train(args.dataset_path)

if __name__ == '__main__':
	main() 