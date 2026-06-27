import os
import json
import uuid
import glob
import shutil
import subprocess
import base64

import yaml as pyyaml
from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    Response,
)
from openai import OpenAI

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

RENDERCV_CMD = os.environ.get("RENDERCV_CMD", "rendercv")

REFERENCE_YAML_PATH = os.path.join(BASE_DIR, "mycv", "me_CV_demo.yaml")
REFERENCE_CV_YAML = ""
FIXED_NON_CV_YAML = ""

if os.path.exists(REFERENCE_YAML_PATH):
    with open(REFERENCE_YAML_PATH, "r", encoding="utf-8") as f:
        ref_data = pyyaml.safe_load(f.read())
    if ref_data and "cv" in ref_data:
        REFERENCE_CV_YAML = pyyaml.dump(
            {"cv": ref_data["cv"]}, allow_unicode=True, default_flow_style=False
        )
    non_cv = {k: v for k, v in ref_data.items() if k != "cv"} if ref_data else {}
    if "settings" in non_cv and isinstance(non_cv["settings"], dict):
        non_cv["settings"].pop("pdf_title", None)
    if non_cv:
        FIXED_NON_CV_YAML = pyyaml.dump(
            non_cv, allow_unicode=True, default_flow_style=False
        )


def _build_non_cv_yaml(cv_yaml: str) -> str:
    if not FIXED_NON_CV_YAML:
        return ""

    cv_name = ""
    try:
        cv_data = pyyaml.safe_load(cv_yaml)
        if isinstance(cv_data, dict) and "cv" in cv_data:
            cv_name = cv_data["cv"].get("name", "")
    except pyyaml.YAMLError:
        pass

    if cv_name:
        non_cv_data = pyyaml.safe_load(FIXED_NON_CV_YAML)
        if "settings" not in non_cv_data:
            non_cv_data["settings"] = {}
        non_cv_data["settings"]["pdf_title"] = f"{cv_name} - 简历"
        return pyyaml.dump(non_cv_data, allow_unicode=True, default_flow_style=False)

    return FIXED_NON_CV_YAML


def _build_full_yaml(cv_yaml: str) -> str:
    non_cv = _build_non_cv_yaml(cv_yaml)
    if non_cv:
        return cv_yaml.rstrip() + "\n\n" + non_cv
    return cv_yaml


# ================================================================== #
#  辅助：执行 rendercv render                                          #
# ================================================================== #
def _run_rendercv(yaml_text: str):
    job_dir = os.path.join(TEMP_DIR, uuid.uuid4().hex[:8])
    os.makedirs(job_dir, exist_ok=True)
    yaml_path = os.path.join(job_dir, "CV.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    try:
        proc = subprocess.run(
            [RENDERCV_CMD, "render", yaml_path],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=job_dir,
        )
    except FileNotFoundError:
        return (
            job_dir,
            None,
            (
                f"'{RENDERCV_CMD}' not found. "
                'Install it with:  pip install "rendercv[full]"'
            ),
        )
    except subprocess.TimeoutExpired:
        return job_dir, None, "Rendering timed out (120 s)."

    pdfs = glob.glob(os.path.join(job_dir, "**", "*.pdf"), recursive=True)

    error = None
    if not pdfs and proc.returncode != 0:
        error = proc.stderr.strip() or proc.stdout.strip() or "Unknown error"

    return job_dir, pdfs[0] if pdfs else None, error


# ================================================================== #
#  路由                                                                #
# ================================================================== #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/preview", methods=["POST"])
def preview():
    data = request.get_json(silent=True) or {}
    yaml_text = data.get("yaml", "")

    if not yaml_text.strip():
        return jsonify(success=False, error="YAML content is empty"), 400

    full_yaml = _build_full_yaml(yaml_text)
    job_dir, pdf_path, error = _run_rendercv(full_yaml)

    try:
        if error:
            return jsonify(success=False, error=error), 400

        if pdf_path:
            with open(pdf_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return jsonify(success=True, content=b64)

        return jsonify(success=False, error="No output files generated."), 500
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


# ================================================================== #
#  AI 智能优化（SSE 流式输出）                                          #
# ================================================================== #
@app.route("/api/optimize", methods=["POST"])
def optimize():
    data = request.get_json(silent=True) or {}
    yaml_text = data.get("yaml", "").strip()
    instruction = data.get("instruction", "").strip()

    if not yaml_text:
        return jsonify(success=False, error="YAML 内容为空"), 400
    if not instruction:
        return jsonify(success=False, error="请输入优化需求"), 400

    api_key = os.environ.get("MIMO_API_KEY")
    if not api_key:
        return jsonify(
            success=False, error="未配置 MIMO_API_KEY 环境变量，请先设置 API Key。"
        ), 500

    base_url = os.environ.get("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
    model = os.environ.get("MIMO_MODEL", "mimo-v2.5-pro")

    system_prompt = f"""你是一位专业的简历/履历工程师和技术写作专家。
你的唯一任务是接收一份 RenderCV YAML 的 cv 部分和用户指令，然后返回优化后的 cv 部分。

## 严格输出规则
1. 只返回优化后的 cv 部分 YAML —— 绝对不要包含 Markdown 围栏（```）、解释、评论或前后缀文本。
2. 输出必须以 `cv:` 开头，是有效的 YAML。
3. 保留所有现有数据，除非用户明确要求更改或删除。
4. 添加新内容时，使用真实但明显的占位符数据，方便用户查找和替换（如"公司名称"、"示例大学"）。
5. 使用 2 空格缩进，不要使用制表符。
6. 日期必须是 YYYY-MM、YYYY-MM-DD 或 "present" 格式。
7. 要点描述应简洁、以行动为导向，并尽可能量化成果。
8. 章节名称保持英文（education、experience、skills、projects 等），因为 RenderCV 需要。

## 参考示例（结构参考）
```yaml
{REFERENCE_CV_YAML}
```
"""

    user_message = f"""## 当前 cv 部分 YAML
```yaml
{yaml_text}
```

## 用户优化需求
{instruction}


请返回完整的优化后 cv 部分 YAML（仅 cv 部分，无围栏）："""

    def generate():
        client = OpenAI(api_key=api_key, base_url=base_url)
        cv_buffer = ""
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.6,
                max_tokens=8192,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    cv_buffer += token
                    yield f"data: {json.dumps({'content': token})}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
