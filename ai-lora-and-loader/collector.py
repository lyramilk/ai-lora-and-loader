import os
import git
from pathlib import Path
import json
from typing import List, Dict
import logging

class GitCodeCollector:
	def __init__(self, base_dir: str):
		self.base_dir = Path(base_dir)
		self.setup_logging()
		
	def setup_logging(self):
		logging.basicConfig(
			level=logging.INFO,
			format='%(asctime)s - %(levelname)s - %(message)s'
		)
		self.logger = logging.getLogger(__name__)
	
	def find_git_repos(self) -> List[Path]:
		"""查找所有git仓库目录"""
		git_repos = []
		for item in self.base_dir.iterdir():
			if item.is_dir() and (item / '.git').exists():
				git_repos.append(item)
		return git_repos
	
	def collect_code_files(self, repo_path: Path) -> List[Dict]:
		"""收集单个仓库中的代码文件"""
		code_files = []
		repo = git.Repo(repo_path)
		
		# 获取所有文件
		for item in repo.tree().traverse():
			if item.type != 'blob':
				continue
			file_path = Path(item.path)
			# 支持的文件类型
			# if file_path.suffix in ['.py', '.cpp', '.c', '.h', '.java', '.js', '.go', '.xml', '.yml', '.md', '.go', '.sh','.proto','.properties','.ini','.yml','.yaml','.json','.toml','.html','.css','.htm','.txt','.sql']:
			if file_path.suffix in ['.py', '.cpp', '.c', '.h', '.java', '.js', '.go', '.xml', '.yml', '.md', '.go', '.sh','.proto']:
				try:
					content = item.data_stream.read().decode('utf-8')
					code_files.append({
						'path': str(item.path),
						'content': content,
						'repo': repo_path.name
					})
				except UnicodeDecodeError:
					self.logger.warning(f"无法解码文件: {item.path}")
					continue
		
		return code_files

	def save_dataset(self, code_files: List[Dict], output_file: str):
		"""保存收集的代码到数据集文件"""
		dataset = []
		
		for code_file in code_files:
			# 构建训练样本格式
			sample = {
				'conversations': [
					{
						'role': 'system',
						'content': '你是一个代码助手。'
					},
					{
						'role': 'human', 
						'content': f'请解释这段代码的功能:\n```{code_file["path"]}\n{code_file["content"]}```'
					},
					{
						'role': 'assistant',
						'content': f'这是来自{code_file["repo"]}仓库的代码文件。让我为您解释其功能。'
					}
				]
			}
			dataset.append(sample)
			
		with open(output_file, 'w', encoding='utf-8') as f:
			json.dump(dataset, f, ensure_ascii=False, indent=2)
		
		self.logger.info(f'数据集已保存到: {output_file}')

	def run(self, output_file: str):
		"""运行代码收集流程"""
		self.logger.info(f'开始扫描目录: {self.base_dir}')
		
		all_code_files = []
		repos = self.find_git_repos()
		
		for repo in repos:
			self.logger.info(f'处理仓库: {repo.name}')
			code_files = self.collect_code_files(repo)
			all_code_files.extend(code_files)
			
		self.logger.info(f'共收集了 {len(all_code_files)} 个代码文件')
		self.save_dataset(all_code_files, output_file) 