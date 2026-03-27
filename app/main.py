from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import asyncio
import base64
import json
import time
from typing import Optional

from generator import ImageGenerator

app = FastAPI(title="NanoBanana2 Image Generator")

DELAY_BETWEEN_IMAGES = 10  # seconds between image generations


@app.post("/api/generate")
async def generate_images(
    api_key: str = Form(...),
    instructions: str = Form(...),
    transparent_bg: str = Form("true"),
    line_stamp_mode: str = Form("false"),
    reference_images: Optional[list[UploadFile]] = File(default=None),
):
    loop = asyncio.get_event_loop()
    is_transparent = transparent_bg.lower() in ("true", "1", "on")
    is_line_stamp = line_stamp_mode.lower() in ("true", "1", "on")

    generator = ImageGenerator(api_key)

    ref_bytes_list: list[bytes] = []
    if reference_images:
        for img_file in reference_images:
            data = await img_file.read()
            if data:
                ref_bytes_list.append(data)

    lines = [line.strip() for line in instructions.split("\u30fb") if line.strip()]
    if not lines:
        async def empty():
            yield json.dumps({"type": "error", "message": "指示が入力されていません"}) + "\n"
        return StreamingResponse(empty(), media_type="application/x-ndjson")

    # Queue for retry status messages from the generator
    retry_queue: asyncio.Queue = asyncio.Queue()

    def on_retry(attempt, max_retries, wait_seconds):
        retry_queue.put_nowait({
            "type": "status",
            "message": f"レート制限 — リトライ中 ({attempt}/{max_retries})、{int(wait_seconds)}秒待機...",
        })

    generator._on_retry = on_retry

    async def stream():
        char_desc = ""
        if ref_bytes_list:
            yield json.dumps({
                "type": "status",
                "message": "参考画像を分析中...",
            }) + "\n"
            try:
                char_desc = await loop.run_in_executor(
                    None, generator.analyze_references, ref_bytes_list,
                )
            except Exception as e:
                yield json.dumps({
                    "type": "error",
                    "message": f"参考画像の分析に失敗: {e}",
                }) + "\n"
                return

            yield json.dumps({
                "type": "status",
                "message": "参考画像の分析完了",
            }) + "\n"

        total = len(lines)
        for i, instruction in enumerate(lines):
            yield json.dumps({
                "type": "progress",
                "current": i + 1,
                "total": total,
                "instruction": instruction,
            }) + "\n"

            try:
                # Optimize prompt
                yield json.dumps({
                    "type": "status",
                    "message": f"プロンプト最適化中... ({i + 1}/{total})",
                }) + "\n"

                optimized = await loop.run_in_executor(
                    None,
                    generator.optimize_prompt,
                    instruction,
                    char_desc,
                    is_transparent,
                    is_line_stamp,
                )

                # Flush any retry messages
                while not retry_queue.empty():
                    yield json.dumps(retry_queue.get_nowait()) + "\n"

                yield json.dumps({
                    "type": "status",
                    "message": f"画像生成中... ({i + 1}/{total})",
                }) + "\n"

                # Generate image
                image_bytes = await loop.run_in_executor(
                    None,
                    generator.generate_image,
                    optimized["prompt"],
                    ref_bytes_list,
                    optimized["aspect_ratio"],
                )

                # Flush any retry messages
                while not retry_queue.empty():
                    yield json.dumps(retry_queue.get_nowait()) + "\n"

                image_b64 = base64.b64encode(image_bytes).decode()

                yield json.dumps({
                    "type": "result",
                    "index": i,
                    "instruction": instruction,
                    "prompt": optimized["prompt"],
                    "aspect_ratio": optimized["aspect_ratio"],
                    "image": image_b64,
                }) + "\n"

            except Exception as e:
                # Flush any retry messages
                while not retry_queue.empty():
                    yield json.dumps(retry_queue.get_nowait()) + "\n"

                yield json.dumps({
                    "type": "error",
                    "index": i,
                    "instruction": instruction,
                    "message": str(e),
                }) + "\n"

            # Delay between images to avoid rate limits
            if i < total - 1:
                yield json.dumps({
                    "type": "status",
                    "message": f"次の生成まで{DELAY_BETWEEN_IMAGES}秒待機中...",
                }) + "\n"
                await asyncio.sleep(DELAY_BETWEEN_IMAGES)

        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


app.mount("/", StaticFiles(directory="static", html=True), name="static")
