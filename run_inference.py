"""Run Spatial4D visual QA inference against an OpenAI-compatible model service."""
import json
import base64
import hashlib
import mimetypes
import re
import os
import tempfile
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import cv2
import numpy as np
from spatial4d.config import RULES_VERSION, NA_QUESTION_TYPES
from spatial4d.data import options_dict, read_qa_rows, task_name
from spatial4d.rules import extract_options, score_case

# ================= 配置区 =================
BASE_JSON_PATH = os.environ.get('SPATIAL4D_QA_DIR', '/data1/wp/code/Spatial4D-Bench/qa_file_shuffle')
MEDIA_BASE_DIR = os.environ.get('SPATIAL4D_MEDIA_DIR', '/data1/wp/code/Spatial4D-Bench/')
# ================= 新增配置 =================
EXTRACTED_FRAMES_DIR = os.environ.get('SPATIAL4D_FRAME_CACHE_DIR', '/data1/wp/code/qwen_model/cache_image')
# ==========================================

VLLM_API_BASE = os.environ.get('SPATIAL4D_API_BASE', 'http://localhost:8000/v1')
MODEL_NAME = os.environ.get('SPATIAL4D_MODEL', 'Qwen/Qwen3.5-397B')
SAVE_FOLDER = os.environ.get('SPATIAL4D_RESULT_DIR', os.path.join('/data1/wp/code/qwen/eval_results/', MODEL_NAME))

DEBUG_LIMIT = -1  # 正数：每个文件只测前N条；-1：完整评测
# ==========================================

# Importing the script must not create directories or initialize a service.
client = None


def extract_and_encode_video(video_path, dataset_name, num_frames=64, target_size=(640, 480)):
    """Uniformly sample real frames; reject missing/corrupt media and caches."""
    if not isinstance(num_frames, int) or num_frames <= 0 or len(target_size) != 2 or min(target_size) <= 0:
        raise ValueError("Frame count and image dimensions must be positive")
    video_path = os.path.abspath(video_path)
    source = Path(video_path).stat()
    video_basename = Path(video_path).stem
    dataset = re.sub(r"[^\w.-]+", "_", str(dataset_name or "unknown")).strip(".") or "unknown"
    # Paths, source changes and extraction settings must not share a cache.
    identity = json.dumps([os.path.normcase(video_path), source.st_size, source.st_mtime_ns,
                           num_frames, target_size], ensure_ascii=False)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    target_dir = os.path.join(EXTRACTED_FRAMES_DIR, dataset, f"{video_basename}_{digest}")
    expected_paths = [os.path.join(target_dir, f"frame_{i:03d}.jpg") for i in range(num_frames)]
    local_image_uris = [Path(p).resolve().as_uri() for p in expected_paths]
    cache_valid = True
    for path in expected_paths:
        if not os.path.isfile(path):
            cache_valid = False
            break
        image = cv2.imread(path)
        if image is None or image.shape[:2] != (target_size[1], target_size[0]):
            cache_valid = False
            break
    if cache_valid:
        return local_image_uris
    os.makedirs(target_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"视频没有可读帧: {video_path}")
        # Short videos repeat real frame indices, not fabricated black images.
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"视频帧解码失败: {video_path}, frame={idx}")
            frame_resized = cv2.resize(frame, target_size)
            if not cv2.imwrite(expected_paths[i], frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 70]):
                raise OSError(f"抽帧写入失败: {expected_paths[i]}")
    finally:
        cap.release()
    return local_image_uris


def clean_options(option_str):
    choices = options_dict(option_str)
    return "\n".join(f"{label}. {text}" for label, text in choices.items()) or None


def encode_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    if not data or cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR) is None:
        raise ValueError(f"无效的图片文件: {image_path}")
    mime = mimetypes.guess_type(str(image_path))[0]
    if not mime or not mime.startswith("image/"):
        raise ValueError(f"无法识别图片类型: {image_path}")
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"


def build_multimodal_message(question, options_raw, image_data, video_data, dataset_name):
    content_list = []
    # A media failure is not a valid text-only spatial evaluation.
    if video_data is not None and str(video_data).strip().casefold() not in {"", "none", "null", "[]"}:
        if not isinstance(video_data, str):
            raise ValueError("video_path must be a path string")
        abs_path = os.path.abspath(os.path.join(MEDIA_BASE_DIR, video_data))
        frames = extract_and_encode_video(abs_path, dataset_name, num_frames=64, target_size=(640, 480))
        for img_uri in frames:
            content_list.append({"type": "image_url", "image_url": {"url": img_uri}})
    elif image_data:
        imgs = image_data if isinstance(image_data, list) else [image_data]
        for p in imgs:
            if p is None or str(p).strip().casefold() in {"", "none", "null"}: continue
            if not isinstance(p, str):
                raise ValueError("image_path must contain path strings")
            uri = encode_image(os.path.join(MEDIA_BASE_DIR, p))
            content_list.append({"type": "image_url", "image_url": {"url": uri}})
    if not content_list:
        raise ValueError("样本没有可用的图片或视频")

    prompt = (
        "You are an expert spatial reasoning assistant.\n"
        "Analyze the visual sequence to construct a 3D cognitive map.\n\n"
        f"Question: {question}\n"
    )

    # 核心逻辑：通过 options 判断题目类型并给出指令
    formatted_options = clean_options(options_raw)

    if formatted_options:
        # 选择题模式
        prompt += f"Options:\n{formatted_options}\n\n"
        prompt += (
            "Instruction: Strictly choose the correct option letter(s) from the list above. "
            "Output only the letter(s) (e.g., 'A' or 'A, B'). No explanations."
        )
    else:
        # 数值题模式
        prompt += (
            "\nInstruction: Calculate the precise value based on the visual information. "
            "Output only the numerical result rounded to one decimal place (e.g., '12.5'). "
            "No units or explanations."
        )

    content_list.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content_list}]


def main():
    global client
    # Validate every QA file before sending any requests. The scorer and
    # inference script deliberately share the same task/options/QA contract.
    if not os.path.isdir(BASE_JSON_PATH):
        raise ValueError(f"QA directory not found: {BASE_JSON_PATH}")
    json_files = sorted(f for f in os.listdir(BASE_JSON_PATH)
                        if f.endswith('.json') and os.path.isfile(os.path.join(BASE_JSON_PATH, f)))
    if not json_files:
        raise ValueError(f"No QA JSON files: {BASE_JSON_PATH}")
    jobs, seen = [], set()
    for file in json_files:
        question_type = task_name(Path(file).stem)
        rows = read_qa_rows(Path(BASE_JSON_PATH) / file)
        if not rows:
            raise ValueError(f"Empty QA file: {file}")
        for row in rows:
            if row.get('id') is None or not isinstance(row.get('question'), str) or not row['question'].strip():
                raise ValueError(f"QA requires id and a nonempty question: {file}")
            if 'ground_truth' not in row:
                raise ValueError(f"QA requires ground_truth: {file}, id={row['id']}")
            if row.get('question_type') and task_name(row['question_type']) != question_type:
                raise ValueError(f"QA task conflicts with filename: {file}, id={row['id']}")
            key = (question_type, str(row['id']))
            if key in seen:
                raise ValueError(f"Duplicate QA task/id: {key}")
            seen.add(key)
            choices = options_dict(row.get('options'))
            if not choices and question_type not in NA_QUESTION_TYPES:
                choices = extract_options(row['question'])
                row['options'] = choices
            if bool(choices) != (question_type not in NA_QUESTION_TYPES):
                raise ValueError(f"Options conflict with task: {file}, id={row['id']}")
            score_case('', row['ground_truth'], question_type, row['question'], choices)
        if DEBUG_LIMIT > 0:
            rows = rows[:DEBUG_LIMIT]
        output = Path(SAVE_FOLDER) / f"result_{file}"
        if output.exists():
            try:
                previous = json.loads(output.read_text(encoding='utf-8-sig'))
                complete = isinstance(previous, list) and len(previous) == len(rows) and all(
                    isinstance(saved, dict) and str(saved.get('id')) == str(row['id'])
                    and saved.get('ground_truth') == row['ground_truth'] and 'model_answer' in saved
                    for saved, row in zip(previous, rows))
            except (ValueError, OSError):
                complete = False
            if not complete:
                raise ValueError(f"Existing result is incomplete or mismatched; preserve/move it before rerunning: {output}")
            print(f"Skipping complete result: {output}")
            continue
        jobs.append((file, question_type, rows, output))
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    for file, question_type, rows, output in jobs:
        print(f"#########processing {file}")
        results = []
        for row in tqdm(rows, desc="Evaluating"):
            inference_status, inference_error = "ok", ""
            try:
                messages = build_multimodal_message(
                    row.get('question'), row.get('options'), row.get('image_path'),
                    row.get('video_path'), row.get('dataset'))
                if client is None:
                    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), base_url=VLLM_API_BASE)
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=32768,
                    temperature=0.7,
                    top_p=0.8,
                    presence_penalty=1.5,
                    extra_body={"mm_processor_kwargs": {"num_frames": 64, "max_pixels": 307200, "min_pixels": 307200},
                    "top_k":20,
                    "chat_template_kwargs": {"enable_thinking": False}}
                )
                ans = (response.choices[0].message.content or "").strip()
            except Exception as e:
                ans = f"Error: {str(e)}"
                inference_status, inference_error = "failed", str(e)

            results.append({
                "id": row['id'],
                "scene": row.get('scene_name'),
                "is_choice": question_type not in NA_QUESTION_TYPES,
                "ground_truth": row['ground_truth'],
                "model_answer": ans,
                "question_type": question_type,
                "question": row.get('question'),
                "options": row.get('options'),
                "dataset": row.get('dataset'),
                "inference_status": inference_status,
                "inference_error": inference_error,
                "rules_version": RULES_VERSION
            })

        # Write on the same filesystem, then atomically replace. A crash must
        # not leave a truncated result that a later run silently skips.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output.parent,
                                             prefix=output.name + '.', suffix='.tmp', delete=False) as f:
                temporary = f.name
                json.dump(results, f, ensure_ascii=False, indent=4)
            os.replace(temporary, output)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)


if __name__ == "__main__":
    main()
