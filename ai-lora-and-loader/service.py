import aiohttp
from aiohttp import web
from aiohttp.web import StreamResponse
import logging
from pathlib import Path
import json
from typing import List, Dict, Optional, AsyncGenerator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import asyncio
import time
from threading import Thread, Event
from transformers import TextIteratorStreamer
import configparser

class StopOnSignal(StoppingCriteria):
	"""停止条件：检查停止信号"""
	def __init__(self, stop_signal: Event):
		self.stop_signal = stop_signal
		
	def __call__(self, input_ids, scores, **kwargs) -> bool:
		return self.stop_signal.is_set()

class ChatModel:
	def __init__(self, model_path: str, tokenizer_path: str = None, is_lora: bool = False):
		"""初始化聊天模型
		Args:
			model_path: 模型路径(LoRA时为权重路径)
			tokenizer_path: 分词器路径(LoRA时为基础模型路径)
			is_lora: 是否为LoRA模型
		"""
		self.setup_logging()
		
		try:
			# 加载tokenizer
			tokenizer_path = tokenizer_path or model_path
			self.tokenizer = AutoTokenizer.from_pretrained(
				tokenizer_path,
				trust_remote_code=True
			)
			
			if is_lora:
				# 加载基础模型
				base_model = AutoModelForCausalLM.from_pretrained(
					tokenizer_path,
					trust_remote_code=True,
					torch_dtype=torch.float16,
					device_map="auto"
				)
				# 加载LoRA权重
				self.model = PeftModel.from_pretrained(
					base_model,
					model_path,
					torch_dtype=torch.float16,
					device_map="auto"
				)
			else:
				# 加载完整模型
				self.model = AutoModelForCausalLM.from_pretrained(
					model_path,
					trust_remote_code=True,
					torch_dtype=torch.float16,
					device_map="auto"
				)
				
			self.logger.info(f"模型加载成功: {model_path}")
			
			# 状态管理
			self.generation_thread = None
			self.stop_signal = Event()
			self.streamer = None
			
		except Exception as e:
			self.logger.error(f"模型加载失败: {e}")
			raise
			
	def setup_logging(self):
		logging.basicConfig(
			level=logging.INFO,
			format='%(asctime)s - %(levelname)s - %(message)s'
		)
		self.logger = logging.getLogger(__name__)
		
	async def stop_generation(self):
		"""停止生成"""
		self.stop_signal.set()
		self.logger.info("收到停止信号")
		if self.streamer:
			self.streamer.stop = True
	
	def _generate(self, **kwargs):
		"""在线程中运行生成任务"""
		try:
			self.model.generate(**kwargs)
		except Exception as e:
			if not self.stop_signal.is_set():
				self.logger.error(f"生成失败: {e}")
				raise
	
	async def chat_completion_stream(self, messages: List[Dict],
								   temperature: float = 0.7,
								   max_tokens: int = 2048) -> AsyncGenerator[Dict, None]:
		"""流式生成回复"""
		try:
			# 重置状态
			self.stop_signal.clear()
			self.streamer = TextIteratorStreamer(
				self.tokenizer,
				skip_special_tokens=False,
				timeout=10.0  # 添加超时
			)
			
			# 构建提示
			prompt = ""
			for msg in messages:
				role = msg.get("role", "user")
				content = msg.get("content", "")
				prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
			prompt += "<|im_start|>assistant\n"
			
			# 准备生成参数
			inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
			generation_kwargs = dict(
				input_ids=inputs["input_ids"],
				attention_mask=inputs["attention_mask"],
				max_new_tokens=max_tokens,
				temperature=temperature,
				do_sample=True,
				streamer=self.streamer,
				pad_token_id=self.tokenizer.pad_token_id,
				eos_token_id=self.tokenizer.eos_token_id,
				stopping_criteria=StoppingCriteriaList([StopOnSignal(self.stop_signal)])
			)
			
			# 启动生成线程
			self.generation_thread = Thread(target=self._generate, kwargs=generation_kwargs)
			self.generation_thread.start()
			
			# 准备响应
			start_time = time.time()
			response_id = f"chatcmpl-{int(start_time*1000)}"
			response_base = {
				"id": response_id,
				"object": "chat.completion.chunk",
				"created": int(start_time),
				"model": "lyramilk-lora",
				"choices": [{
					"index": 0,
					"delta": {},
					"finish_reason": None
				}]
			}
			
			# 发送role
			yield {**response_base, "choices": [{
				"index": 0,
				"delta": {"role": "assistant"},
				"finish_reason": None
			}]}
			
			# 处理生成的文本
			started = False
			async for text in self._stream_text():
				if "<|im_start|>assistant" in text:
					started = True
					text = text.split("<|im_start|>assistant")[-1].strip()
					if not text:
						continue
				
				if "<|im_end|>" in text:
					text = text.split("<|im_end|>")[0].strip()
					if text:
						yield {**response_base, "choices": [{
							"index": 0,
							"delta": {"content": text},
							"finish_reason": None
						}]}
					break
				
				if started and text.strip():
					yield {**response_base, "choices": [{
						"index": 0,
						"delta": {"content": text},
						"finish_reason": None
					}]}
			
			# 发送结束标记
			yield {**response_base, "choices": [{
				"index": 0,
				"delta": {},
				"finish_reason": "stop" if not self.stop_signal.is_set() else "user_cancel"
			}]}
			
		except Exception as e:
			self.logger.error(f"生成失败: {e}")
			yield {**response_base, "choices": [{
				"index": 0,
				"delta": {"content": f"\n\n生成失败: {str(e)}"},
				"finish_reason": "error"
			}]}
		
		finally:
			self.stop_signal.clear()
			self.streamer = None
			if self.generation_thread and self.generation_thread.is_alive():
				self.generation_thread.join(timeout=1.0)
			self.generation_thread = None
	
	async def _stream_text(self):
		"""异步迭代streamer的输出"""
		while True:
			try:
				text = await asyncio.get_event_loop().run_in_executor(
					None, 
					next, 
					self.streamer
				)
				yield text
			except StopIteration:
				break
			except Exception as e:
				if not self.stop_signal.is_set():
					self.logger.error(f"流式输出错误: {e}")
				break

class ChatServer:
	def __init__(self, model: ChatModel):
		self.model = model
		
	async def chat_completion(self, request: web.Request) -> web.Response:
		"""聊天补全接口"""
		try:
			data = await request.json()
			messages = data.get("messages", [])
			temperature = data.get("temperature", 0.7)
			max_tokens = data.get("max_tokens", 2048)
			stream = data.get("stream", False)
			
			if not messages:
				return web.json_response(
					{"error": "Missing messages"},
					status=400
				)
				
			if stream:
				response = web.StreamResponse(
					status=200,
					reason='OK',
					headers={
						'Content-Type': 'text/event-stream',
						'Cache-Control': 'no-cache',
						'Connection': 'keep-alive',
					}
				)
				await response.prepare(request)
				
				async for chunk in self.model.chat_completion_stream(
					messages, temperature, max_tokens
				):
					await response.write(
						f'data: {json.dumps(chunk)}\n\n'.encode('utf-8')
					)
				
				return response
				
		except json.JSONDecodeError:
			return web.json_response({"error": "Invalid JSON"}, status=400)
			
		except Exception as e:
			return web.json_response({"error": str(e)}, status=500)

	async def stop_generation(self, request: web.Request) -> web.Response:
		"""停止生成接口"""
		await self.model.stop_generation()
		return web.json_response({"success": True})

async def init_app(model_path: str, base_model_path: str = None) -> web.Application:
	"""初始化应用
	Args:
		model_path: 模型路径(LoRA时为权重路径)
		base_model_path: 基础模型路径(仅LoRA时需要)
	"""
	is_lora = base_model_path is not None
	model = ChatModel(model_path, base_model_path, is_lora)
	server = ChatServer(model)
	
	app = web.Application()
	app.router.add_post("/v1/chat/completions", server.chat_completion)
	app.router.add_post("/v1/chat/stop", server.stop_generation)  # 添加停止接口
	
	# 添加静态文件服务
	app.router.add_static('/static/', Path(__file__).parent / 'static')
	
	# 添加根路由重定向
	async def index_handler(request):
		raise web.HTTPFound('/static/index.html')
	
	app.router.add_get('/', index_handler)
	
	return app

def main():
	import argparse
	import configparser
	from pathlib import Path
	
	# 加载配置文件
	config = configparser.ConfigParser()
	config_path = Path(__file__).parent / 'config.ini'
	
	if not config_path.exists():
		raise FileNotFoundError(f"配置文件不存在: {config_path}")
		
	config.read(config_path, encoding='utf-8')
	
	# 解析命令行参数
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", required=False, help="基础模型路径", 
		default=config.get('paths', 'model_path'))
	parser.add_argument("--lora", help="LoRA权重路径，不指定则使用基础模型",
		default=config.get('paths', 'output_dir'))
	parser.add_argument("--port", type=int, default=8080, help="服务端口")
	args = parser.parse_args()
	
	# 如果指定了lora参数，使用LoRA模式
	if args.lora:
		app = asyncio.run(init_app(args.lora, args.model))  # lora权重路径在前，基础模型路径在后
	else:
		app = asyncio.run(init_app(args.model))  # 只使用基础模型
	
	web.run_app(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
	main()
