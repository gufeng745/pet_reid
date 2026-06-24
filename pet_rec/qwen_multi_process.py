import os
import re
import io
import cv2
import json
import time
import base64
import shutil
import requests
import tqdm
import numpy as np
import csv
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import natsort
from openai import OpenAI
import argparse

MAX_WORKER = 2  # 降低并发数，避免 429 限流
NEED_RESIZE = 1
API_TIMEOUT = 120  # API 超时时间（秒），降低从 120 到 60
MAX_RETRIES = 2  # 最大重试次数
BATCH_SIZE = 50  # 每批处理的图片数量，避免一次性加载过多图片到内存
# 打印锁，确保多线程环境下输出不被交错
print_lock = threading.Lock()
# 断点续存相关配置
CHECKPOINT_ENABLED = True  # 是否启用断点续存
CHECKPOINT_PATH = r"D:\claude_workspace\pet_rec\DogsVsCats_dogs-vs-cats-redux-kernels-edition\train\checkpoint.json"  # checkpoint 文件路径

# USER_KEY = "sk-3affaa955c3e481bb1821d5e945e3982"  # 填写https://llmapp.tp-link.com.cn/网站获得的 api 密钥
# USER_KEY = "sk-91ef9ed1e8e348be89ce254f9a7f22a3"
USER_KEY = "tp-c1vfjwno5xf3y684ocm2zdopuwsesssvi7l0ibgb7adqlyi0"
USER_URL = "https://token-plan-cn.xiaomimimo.com/v1"
MODEL_NAME = "mimo-v2.5"

PROMPT_PATH = r'D:\claude_workspace\pet_rec\prompt_pet.txt'  # 提示词路径
TEST_IMG_DIR = r'D:\claude_workspace\pet_rec\DogsVsCats_dogs-vs-cats-redux-kernels-edition\train'
SAVE_DIR = r'D:\claude_workspace\pet_rec\DogsVsCats_dogs-vs-cats-redux-kernels-edition'  # 结果保存文件夹


class CheckpointManager:
    """断点续存管理器，用于保存和恢复处理进度"""
    
    def __init__(self, checkpoint_path: str, enabled: bool = True):
        self.checkpoint_path = checkpoint_path
        self.enabled = enabled
        self.data = {
            "processed_images": {},
            "statistics": [0, 0, 0, 0],
            "total_images": 0,
            "completed_count": 0
        }
        self.lock = threading.Lock()
        
        if enabled:
            self.load_checkpoint()
    
    def load_checkpoint(self):
        """加载之前的 checkpoint 文件"""
        if not self.enabled:
            return
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                print(f"[Checkpoint] 已加载 checkpoint 文件：{self.checkpoint_path}")
                print(f"[Checkpoint] 已处理 {self.data['completed_count']}/{self.data['total_images']} 张图片")
            except Exception as e:
                print(f"[Checkpoint] 加载 checkpoint 失败，重新开始：{e}")
                self.data = {
                    "processed_images": {},
                    "statistics": [0, 0, 0, 0],
                    "total_images": 0,
                    "completed_count": 0
                }
        else:
            print(f"[Checkpoint] 未找到 checkpoint 文件，开始新任务")
    
    def _save_checkpoint_internal(self):
        """内部方法：保存 checkpoint 到文件（调用时必须已持有锁）"""
        try:
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            print(f"[Checkpoint] 已保存 checkpoint (进度：{self.data['completed_count']}/{self.data['total_images']})")
        except Exception as e:
            print(f"[Checkpoint] 保存 checkpoint 失败：{e}")
    
    def save_checkpoint(self):
        """保存当前进度到 checkpoint 文件"""
        if not self.enabled:
            return
        with self.lock:
            self._save_checkpoint_internal()
    
    def mark_completed(self, img_name: str, answer: Dict, answer_raw: str, stat_type: int, force_save: bool = False):
        """标记某张图片处理完成
        Args:
            img_name: 图片名称
            answer: 解析后的答案
            answer_raw: 原始答案
            stat_type: 统计类型 (0: unavailable, 1: unqualified, 2: qualified, 3: others)
            force_save: 是否强制保存 checkpoint（用于最后保存）
        """
        if not self.enabled:
            return
        should_save = False
        with self.lock:
            self.data["processed_images"][img_name] = {
                "status": "success",
                "answer": answer,
                "answer_raw": answer_raw,
                "stat_type": stat_type,
                "timestamp": time.time()
            }
            self.data["completed_count"] += 1
            # 更新统计数据
            self.data["statistics"][stat_type] += 1
            # 每处理 10 张时保存，或强制保存
            if force_save or self.data["completed_count"] % 10 == 0:
                should_save = True
        # 释放锁后再保存，避免持有锁时进行 I/O 操作
        if should_save:
            self.save_checkpoint()
    
    def mark_failed(self, img_name: str, error: str):
        """标记某张图片处理失败
        注意：失败时不立即保存 checkpoint，避免频繁 I/O 影响性能。
        定期保存由 mark_completed 方法中的每 10 张保存机制保证。
        """
        if not self.enabled:
            return
        with self.lock:
            self.data["processed_images"][img_name] = {
                "status": "fail",
                "error": str(error),
                "timestamp": time.time()
            }
            self.data["completed_count"] += 1
            # 失败也计入 unavailable (stat_type=0)
            self.data["statistics"][0] += 1
        # 失败时不立即保存，避免频繁 I/O
        # checkpoint 会在每处理 10 张图片或最后统一保存
    
    def is_processed(self, img_name: str) -> bool:
        """检查图片是否已处理"""
        if not self.enabled:
            return False
        with self.lock:
            return img_name in self.data["processed_images"]
    
    def get_processed_list(self) -> List[str]:
        """获取已处理图片列表"""
        if not self.enabled:
            return []
        with self.lock:
            return list(self.data["processed_images"].keys())
    
    def get_successful_results(self) -> List[Dict]:
        """获取所有成功的处理结果"""
        if not self.enabled:
            return []
        with self.lock:
            results = []
            for img_name, info in self.data["processed_images"].items():
                if info.get("status") == "success":
                    results.append({
                        "name": img_name,
                        "answer": info.get("answer"),
                        "answer_raw": info.get("answer_raw"),
                        "stat_type": info.get("stat_type", 0)
                    })
            return results
    
    def get_statistics(self) -> List[int]:
        """获取统计数据"""
        with self.lock:
            return self.data["statistics"].copy()
    
    def set_total(self, total: int):
        """设置总图片数"""
        with self.lock:
            self.data["total_images"] = total
    
    def reset(self):
        """重置 checkpoint（清空所有记录）"""
        with self.lock:
            self.data = {
                "processed_images": {},
                "statistics": [0, 0, 0, 0],
                "total_images": 0,
                "completed_count": 0
            }
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
        print("[Checkpoint] 已重置 checkpoint")


# 输出结果是 QUALIFIED/UNQUALIFIED/UNAVAILABLE
def cv2_imread(file_path: str):
    """使用 cv2 读取图片，兼容中文路径。"""
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)


def pil_imwrite(file_path: str, img: np.ndarray):
    """使用 Pillow 保存图像。"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_pil.save(file_path)


def encode_image(image_path: str) -> str:
    """将图片编码为 base64 字符串。"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_file_if_not_exists(file_path: str):
    """如果文件不存在则创建空文件。"""
    path = Path(file_path)
    if not path.exists():
        path.touch()
        print(f"文件已创建：{file_path}")
    else:
        print(f"文件已存在：{file_path}")


def parse_answer(answer: Any) -> Dict:
    """解析模型响应，兼容直接 json 和```json```包裹的情况，返回 dict。"""
    if isinstance(answer, str):
        # 先尝试用正则表达式提取 json 内容
        pattern = r'```json\s*([\s\S]*?)\s*```'
        match = re.search(pattern, answer, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            return json.loads(json_str)
        # 如果正则匹配失败，尝试直接移除标记
        cleaned = answer.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        if cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        return json.loads(cleaned)
    elif isinstance(answer, dict):
        return answer
    else:
        raise ValueError("answer 格式无法解析")


def draw_and_save_image(images_path: str, images_name: str, answer: Dict, dest_path: str):
    """绘制检测框并保存图片。"""
    img_path = os.path.join(images_path, images_name)
    img = cv2_imread(img_path)
    if img is None:
        print(f"图片读取失败：{img_path}")
        return
    if NEED_RESIZE:
        img = cv2.resize(img, (1280, 720))
    # 兼容字段名
    locations = answer.get('locations', answer.get('location', []))
    for coord in locations:
        if not coord:
            continue
        color = [0, 0, 255] if int(coord[-1]) == 1 else [255, 0, 0]
        cv2.rectangle(img, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), color=color, thickness=2)
    os.makedirs(dest_path, exist_ok=True)
    pil_imwrite(os.path.join(dest_path, images_name), img)


def stat_result_by_results_field(answer: Dict) -> int:
    """
    根据 answer 中的'results'字段统计:
    返回 0: unavailable, 1: unqualified, 2: qualified, 3: others
    """
    # 兼容字段名
    results = answer.get('results', None)
    if results is None:
        return 3  # others
    results_upper = str(results).strip().upper()
    if results_upper == "UNAVAILABLE":
        return 0
    elif results_upper == "UNQUALIFIED":
        return 1
    elif results_upper == "QUALIFIED":
        return 2
    else:
        return 3


def call_openai_api(user_key, user_url, model_name, prompt, image_base64, system_prompt, retries=0):
    """通过 OpenAI 客户端调用 API，带重试机制"""
    client = OpenAI(
        api_key=user_key,
        base_url=user_url,
        timeout=API_TIMEOUT  # 设置 API 超时时间
    )
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.1,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        # 兼容 OpenAI 和百炼返回格式
        if hasattr(response, 'choices') and response.choices:
            return response.choices[0].message.content
        elif hasattr(response, 'output') and response.output:
            return response.output.choices[0].message.content
        else:
            raise RuntimeError("API 返回格式异常：" + str(response))
    except Exception as e:
        if retries < MAX_RETRIES:
            with print_lock:
                print(f"  [重试 {retries+1}/{MAX_RETRIES}] 请求失败，正在重试...")
            # 429 错误需要等待更长时间
            if '429' in str(e):
                wait_time = 5 * (retries + 1)  # 递增等待时间
                with print_lock:
                    print(f"  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                time.sleep(2)
            return call_openai_api(user_key, user_url, model_name, prompt, image_base64, system_prompt, retries + 1)
        raise RuntimeError(f"API 请求失败（已重试{MAX_RETRIES}次）: {e}")


# def parse_animals_to_csv_rows(img_name: str, answer: Dict) -> List[Dict]:
#     """
#     解析 LLM 返回的 animals 数据，转换为 CSV 行格式。
#     返回包含多个宠物记录的列表（一张图片可能有多个宠物）。
#     """
#     rows = []
#     if not answer or not isinstance(answer, dict):
#         return rows
    
#     animals = answer.get("animals", [])
#     if not animals:
#         return rows
    
#     for animal in animals:
#         if not isinstance(animal, dict):
#             continue
#         animal_type = animal.get("type", "unknown")
#         attributes = animal.get("attributes", {})
#         if not isinstance(attributes, dict):
#             attributes = {}
        
#         row = {
#             "filename": img_name,
#             "type": animal_type,
#             "breed": attributes.get("breed", "unknown"),
#             "color": attributes.get("color", "unknown"),
#             "pattern": attributes.get("pattern", "unknown"),
#             "body_size": attributes.get("body_size", "unknown")
#         }
#         rows.append(row)
    
#     return rows

# 分隔符定义
COLOR_SEPARATOR = ","      # 颜色用逗号分隔
PATTERN_SEPARATOR = "|"    # 花纹用竖线分隔

def parse_animals_to_csv_rows(img_name: str, answer: Dict) -> List[Dict]:
    """
    解析 LLM 返回的 animals 数据，转换为 CSV 行格式。
    返回包含多个宠物记录的列表（一张图片可能有多个宠物）。
    
    CSV 格式：filename,type,breed,color_primary,color_secondary,pattern,body_size
    - color_secondary: 多个颜色用逗号分隔，如 "white,orange"
    - pattern: 多个花纹用竖线分隔，如 "tabby|bicolor"
    """
    rows = []
    if not answer or not isinstance(answer, dict):
        return rows
    
    animals = answer.get("animals", [])
    if not animals:
        return rows
    
    for animal in animals:
        if not isinstance(animal, dict):
            continue
        
        animal_type = animal.get("type", "unknown")
        attributes = animal.get("attributes", {})
        if not isinstance(attributes, dict):
            attributes = {}
        
        # 处理颜色：支持 color_primary + color_secondary 格式
        color_primary = attributes.get("color_primary", "")
        color_secondary = attributes.get("color_secondary", "")
        
        # 向后兼容：如果存在旧的 color 字段
        if not color_primary and "color" in attributes:
            color_primary = attributes.get("color", "unknown")
        
        # 处理花纹：支持多标签（使用竖线分隔）
        pattern = attributes.get("pattern", "unknown")
        
        # 如果 pattern 是列表格式，转换为竖线分隔字符串
        if isinstance(pattern, list):
            pattern = PATTERN_SEPARATOR.join(pattern)
        
        # 如果 color_secondary 是列表格式，转换为逗号分隔字符串
        if isinstance(color_secondary, list):
            color_secondary = COLOR_SEPARATOR.join(color_secondary)
        
        row = {
            "filename": img_name,
            "type": animal_type,
            "breed": attributes.get("breed", "unknown"),
            "color_primary": color_primary if color_primary else "unknown",
            "color_secondary": color_secondary if color_secondary else "",  # 辅颜色可以为空
            "pattern": pattern if pattern else "unknown",
            "body_size": attributes.get("body_size", "unknown")
        }
        rows.append(row)
    
    return rows


# 辅助函数：解析 CSV 中的多标签字段
def parse_colors(color_str: str) -> List[str]:
    """解析颜色字符串为列表"""
    if not color_str or color_str == "" or color_str == "unknown":
        return []
    return [c.strip() for c in color_str.split(COLOR_SEPARATOR) if c.strip()]

def parse_patterns(pattern_str: str) -> List[str]:
    """解析花纹字符串为列表"""
    if not pattern_str or pattern_str == "unknown":
        return []
    return [p.strip() for p in pattern_str.split(PATTERN_SEPARATOR) if p.strip()]




def process_single_image(
        user_key: str,
        user_url: str,
        model_name: str,
        user_prompt: str,
        image: bytes,
        name: str,
        test_img_path: str,
        dest_path: str,
        system_prompt: str,
) -> Dict:
    """单张图片推理任务，用于并发。返回统计类型和 answer。"""
    start_time = time.time()
    try:
        if isinstance(image, bytes):
            decoded_image = image.decode("utf-8")
        else:
            decoded_image = image  # 已经是 str
        answer_raw = call_openai_api(
            user_key=user_key,
            user_url=user_url,
            model_name=model_name,
            prompt=user_prompt,
            image_base64=decoded_image,
            system_prompt=system_prompt
        )
        with print_lock:
            print(f"\n{'='*50}")
            print(f"img_name: {name}")
            print(f"answer: {answer_raw[:200]}..." if answer_raw and len(answer_raw) > 200 else f"answer: {answer_raw}")
            print(f"{'='*50}\n")
        answer = parse_answer(answer_raw)
        # 统计类型
        stat_type = stat_result_by_results_field(answer)
        # 画框保存
        # draw_and_save_image(test_img_path, name, answer, dest_path)
        status = "success"
    except Exception as e:
        with print_lock:
            print(f"\n{'='*50}")
            print(f"img_name: {name}")
            print(f"API 调用失败：{e}")
            print(f"{'='*50}\n")
        answer_raw = None
        answer = None
        stat_type = 0  # unavailable
        status = "fail"
    end_time = time.time()
    with print_lock:
        print(f"处理时间：{end_time - start_time:.2f} 秒")
    # 添加延迟，避免 429 限流
    time.sleep(1)
    return {"name": name, "answer": answer, "answer_raw": answer_raw, "stat_type": stat_type, "status": status}


def table_process_ali(
        user_key: str,
        user_url: str,
        model_name: str,
        encoded_images: List[bytes],
        init_img: str,
        img_name_list: List[str],
        prompt_path: str,
        statistical_data: List[int],
        test_img_path: str,
        dest_path: str,
        csv_output_path: str,
        checkpoint_manager: CheckpointManager = None
):
    """
    批量推理图片并处理结果（并发版）。
    统计数据：[unavailable, unqualified, qualified, others]
    返回：(task_response_list, all_results) - all_results 包含所有处理结果用于写入 CSV
    
    Args:
        checkpoint_manager: CheckpointManager 实例，用于断点续存
    """
    max_workers = MAX_WORKER

    # 系统角色设定
    system_prompt = (
        "你是一个忠实于用户输入图片的监控录像分析专家，擅长准确理解输入图片的内容，并遵循用户的指示，为监控摄像头的购买者提供信息分析服务，忠实于用户的输入视频或图像，不要脱离视频或图像内容作答，减少幻觉。"
    )
    with open(prompt_path, 'r', encoding='utf-8') as f:
        user_prompt = f.read()

    task_response_list = []
    all_results = []  # 保存所有结果用于写入 CSV

    # 统计数据线程安全
    stat_lock = threading.Lock()
    results_lock = threading.Lock()

    # 如果启用了 checkpoint，先加载已成功的结果
    if checkpoint_manager and checkpoint_manager.enabled:
        # 从 checkpoint 加载已成功的结果
        successful_results = checkpoint_manager.get_successful_results()
        for result_data in successful_results:
            all_results.append({
                "name": result_data["name"],
                "answer": result_data["answer"],
                "answer_raw": result_data["answer_raw"],
                "stat_type": result_data["stat_type"],
                "status": "success"
            })
            task_response_list.append(result_data["answer"])
        
        # 更新统计数据
        checkpoint_stats = checkpoint_manager.get_statistics()
        for i, stat in enumerate(checkpoint_stats):
            statistical_data[i] = stat
        
        # 获取已处理的图片列表
        processed_list = checkpoint_manager.get_processed_list()
        print(f"[Checkpoint] 已跳过 {len(processed_list)} 张已处理的图片")
        
        # 只处理未处理的图片
        images_to_process = []
        names_to_process = []
        for image, name in zip(encoded_images, img_name_list):
            if name not in processed_list:
                images_to_process.append(image)
                names_to_process.append(name)
        
        encoded_images = images_to_process
        img_name_list = names_to_process
        
        if not encoded_images:
            print("[Checkpoint] 所有图片已处理完成！")
            print("****************************** llm_forward end ******************************")
            return task_response_list, all_results

    # CSV 表头定义（不写入，只在 main() 中初始化一次）
    fieldnames = ["filename", "type", "breed", "color_primary", "color_secondary", "pattern", "body_size"]

    def update_stat(stat_type: int):
        with stat_lock:
            if stat_type == 0:
                statistical_data[0] += 1  # unavailable
            elif stat_type == 1:
                statistical_data[1] += 1  # unqualified
            elif stat_type == 2:
                statistical_data[2] += 1  # qualified
            elif stat_type == 3:
                statistical_data[3] += 1  # others

    def write_result_to_csv(result: Dict):
        """将单个结果写入 CSV 文件"""
        img_name = result.get("name", "")
        answer = result.get("answer")
        rows = parse_animals_to_csv_rows(img_name, answer)
        
        # 直接写入文件，不需要锁（因为是在主线程顺序执行）
        with open(csv_output_path, 'a', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerows(rows)
            csvfile.flush()  # 立即刷新到磁盘

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for image, name in zip(encoded_images, img_name_list):
            futures.append(
                executor.submit(
                    process_single_image,
                    user_key, user_url, model_name, user_prompt, image, name, test_img_path, dest_path, system_prompt
                )
            )
        for f in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = f.result()
            update_stat(result["stat_type"])
            with results_lock:
                all_results.append(result)
            
            # 更新 checkpoint
            if checkpoint_manager and checkpoint_manager.enabled:
                if result["status"] == "success":
                    checkpoint_manager.mark_completed(
                        result["name"],
                        result["answer"],
                        result["answer_raw"],
                        result["stat_type"]
                    )
                else:
                    checkpoint_manager.mark_failed(result["name"], "API error")
            
            if result["status"] == "success":
                task_response_list.append(result["answer"])
                # 每处理一张图片就立即写入 CSV
                write_result_to_csv(result)
            # 失败的/unavailable 的也统计

    # 最后保存一次 checkpoint
    if checkpoint_manager and checkpoint_manager.enabled:
        checkpoint_manager.save_checkpoint()

    print("****************************** llm_forward end ******************************")
    return task_response_list, all_results


# ================= 主流程 =================

def encode_single_image(img_path: str) -> str:
    """
    编码单张图片为 base64 字符串。
    优化：避免拖影问题，确保每次都新建 BytesIO 对象并且 base64 编码为字符串
    """
    with Image.open(img_path) as img_obj:
        if NEED_RESIZE:
            img_resized = img_obj.resize((1280, 720))
        else:
            img_resized = img_obj
        img_byte_arr = io.BytesIO()
        img_format = img_obj.format if img_obj.format else "PNG"
        img_resized.save(img_byte_arr, format=img_format)
        img_byte_arr.seek(0)
        return base64.b64encode(img_byte_arr.read()).decode("utf-8")


def write_annotations_csv(all_results: List[Dict], output_path: str):
    """
    将所有处理结果写入 annotations.csv 文件。
    列：filename, type, breed, color, pattern, body_size
    """
    fieldnames = ["filename", "type", "breed", "color", "pattern", "body_size"]
    
    all_rows = []
    for result in all_results:
        img_name = result.get("name", "")
        answer = result.get("answer")
        rows = parse_animals_to_csv_rows(img_name, answer)
        all_rows.extend(rows)
    
    # 按文件名排序
    all_rows.sort(key=lambda x: x["filename"])
    
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"CSV 文件已保存：{output_path}")
    print(f"共写入 {len(all_rows)} 条宠物记录")


def main():
    # 大模型基础信息
    user_key = USER_KEY
    user_url = USER_URL
    model_name = MODEL_NAME

    prompt_txt = PROMPT_PATH
    test_img_dir = TEST_IMG_DIR
    dest_path = SAVE_DIR

    os.makedirs(dest_path, exist_ok=True)

    statistical_data = [0, 0, 0, 0]  # [unavailable, unqualified, qualified, others]
    test_img_list = natsort.natsorted(os.listdir(test_img_dir))
    total_images = len(test_img_list)
    
    print(f"共检测到 {total_images} 张图片")
    print(f"每批处理 {BATCH_SIZE} 张图片")

    # CSV 输出路径
    csv_output_path = os.path.join(dest_path, "annotations.csv")

    # 创建 checkpoint 管理器
    checkpoint_manager = CheckpointManager(CHECKPOINT_PATH, CHECKPOINT_ENABLED)
    checkpoint_manager.set_total(total_images)

    # 获取已处理的图片列表
    processed_list = checkpoint_manager.get_processed_list()
    processed_set = set(processed_list)
    
    # 从 checkpoint 恢复统计数据
    checkpoint_stats = checkpoint_manager.get_statistics()
    for i, stat in enumerate(checkpoint_stats):
        statistical_data[i] = stat
    
    # 初始化 CSV 文件（写入表头）
    fieldnames = ["filename", "type", "breed", "color_primary", "color_secondary", "pattern", "body_size"]
    with open(csv_output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # 计算需要处理的图片
    remaining_images = [img for img in test_img_list if img not in processed_set]
    print(f"已处理 {len(processed_set)} 张，剩余 {len(remaining_images)} 张需要处理")
    
    if not remaining_images:
        print("所有图片已处理完成！")
        print("***************最终统计结果****************")
        print(f'UNAVAILABLE count: {statistical_data[0]}')
        print(f'UNQUALIFIED count: {statistical_data[1]}')
        print(f'QUALIFIED count: {statistical_data[2]}')
        print(f'others count: {statistical_data[3]}')
        print("******************************************")
        return

    # 分批处理图片
    batch_count = 0
    all_results = []
    
    for batch_start in range(0, len(remaining_images), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(remaining_images))
        batch_images = remaining_images[batch_start:batch_end]
        batch_count += 1
        
        print(f"\n{'='*60}")
        print(f"批次 {batch_count}: 处理第 {batch_start + 1} - {batch_end} 张图片 (共 {len(batch_images)} 张)")
        print(f"{'='*60}")
        
        # 编码当前批次的图片
        print(f"正在编码批次 {batch_count} 的图片...")
        input_list = []
        for img in batch_images:
            img_path = os.path.join(test_img_dir, img)
            encoded_img = encode_single_image(img_path)
            input_list.append(encoded_img)
        
        # 获取第一张图片作为 init_img（如果有的话）
        init_img = input_list[0] if input_list else ""
        
        # 处理当前批次
        response_list, batch_results = table_process_ali(
            user_key, user_url, model_name, 
            input_list, init_img, batch_images, 
            prompt_txt, statistical_data, test_img_dir,
            dest_path, csv_output_path, checkpoint_manager
        )
        
        all_results.extend(batch_results)
        
        # 清理内存：让当前批次的图片数据可以被垃圾回收
        del input_list
        del batch_images
        
        print(f"批次 {batch_count} 处理完成")

    # 最后保存一次 checkpoint
    if checkpoint_manager and checkpoint_manager.enabled:
        checkpoint_manager.save_checkpoint()

    print("***************最终统计结果****************")
    print(f'UNAVAILABLE count: {statistical_data[0]}')
    print(f'UNQUALIFIED count: {statistical_data[1]}')
    print(f'QUALIFIED count: {statistical_data[2]}')
    print(f'others count: {statistical_data[3]}')
    print("******************************************")
    print(f"CSV 文件已保存：{csv_output_path}")


if __name__ == '__main__':
    main()