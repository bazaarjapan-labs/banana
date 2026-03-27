from google import genai
from google.genai import types
from PIL import Image
import io
import base64
import json
import re
import time


class ImageGenerator:
    IMAGE_MODEL = "gemini-3.1-flash-image-preview"
    TEXT_MODEL = "gemini-2.5-flash"

    VALID_RATIOS = [
        "1:1", "2:3", "3:2", "3:4", "4:3",
        "4:5", "5:4", "9:16", "16:9",
    ]

    MAX_RETRIES = 5

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self._on_retry = None  # callback for retry status

    @staticmethod
    def _friendly_error(e: Exception) -> str:
        msg = str(e)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            return "APIレート制限です。自動リトライでも解消しませんでした。少し時間を置いて再試行してください。"
        if "INVALID_ARGUMENT" in msg or "400" in msg:
            return f"リクエストが不正です: {msg[:200]}"
        if "PERMISSION_DENIED" in msg or "403" in msg:
            return "API Keyが無効、または権限がありません。キーを確認してください。"
        if "NOT_FOUND" in msg or "404" in msg:
            return "モデルが見つかりません。API Keyのプロジェクト設定を確認してください。"
        return f"生成エラー: {msg[:300]}"

    @staticmethod
    def _parse_retry_delay(error_msg: str) -> float:
        """Extract retry delay from error message like 'retry in 17.86s'."""
        match = re.search(r"retry in ([\d.]+)", error_msg)
        if match:
            return min(float(match.group(1)) + 3, 90)
        return 30

    def _call_with_retry(self, fn, *args, **kwargs):
        for attempt in range(self.MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                msg = str(e)
                is_rate_limit = "429" in msg or "RESOURCE_EXHAUSTED" in msg

                if is_rate_limit and attempt < self.MAX_RETRIES - 1:
                    wait = self._parse_retry_delay(msg)
                    if self._on_retry:
                        self._on_retry(attempt + 1, self.MAX_RETRIES, wait)
                    time.sleep(wait)
                    continue
                raise RuntimeError(self._friendly_error(e)) from e

    def _bytes_to_pil(self, image_bytes: bytes) -> Image.Image:
        return Image.open(io.BytesIO(image_bytes))

    @staticmethod
    def _extract_text(response) -> str | None:
        """Safely extract text from a Gemini response."""
        try:
            if response.text:
                return response.text
        except (AttributeError, ValueError):
            pass
        if not getattr(response, "candidates", None):
            return None
        for candidate in response.candidates:
            content = getattr(candidate, "content", None)
            if not content or not getattr(content, "parts", None):
                continue
            for part in content.parts:
                if getattr(part, "text", None):
                    return part.text
        return None

    @staticmethod
    def _extract_image(response) -> bytes | None:
        """Safely extract image bytes from a Gemini response."""
        if not getattr(response, "candidates", None):
            return None
        for candidate in response.candidates:
            content = getattr(candidate, "content", None)
            if not content or not getattr(content, "parts", None):
                continue
            for part in content.parts:
                if getattr(part, "inline_data", None) is not None:
                    data = part.inline_data.data
                    if isinstance(data, str):
                        return base64.b64decode(data)
                    return data
        return None

    def analyze_references(self, images: list[bytes]) -> str:
        if not images:
            return ""
        contents: list = [
            "Analyze these character reference images in detail. Describe:\n"
            "1. Art style (anime, manga, semi-realistic, etc.)\n"
            "2. Hair: color, length, style\n"
            "3. Eyes: color, shape, size\n"
            "4. Face: shape, distinctive features\n"
            "5. Body type and proportions\n"
            "6. Default clothing/outfit\n"
            "7. Accessories, markings, or other distinctive features\n\n"
            "Provide a concise but comprehensive English description for maintaining "
            "character consistency in image generation prompts.",
        ]
        for img_bytes in images:
            contents.append(self._bytes_to_pil(img_bytes))

        response = self._call_with_retry(
            self.client.models.generate_content,
            model=self.TEXT_MODEL,
            contents=contents,
        )
        return self._extract_text(response) or ""

    def optimize_prompt(
        self,
        instruction: str,
        character_desc: str,
        transparent_bg: bool,
        line_stamp_mode: bool,
    ) -> dict:
        quoted_texts = re.findall(r"\u300c(.+?)\u300d", instruction)

        text_rule = ""
        if quoted_texts:
            joined = ", ".join(f'"{t}"' for t in quoted_texts)
            text_rule = (
                f"\n- Render the following text prominently and legibly in the image: {joined}"
            )

        bg_rule = ""
        if transparent_bg:
            bg_rule = (
                "\n- 背景は真っ白(pure white #FFFFFF)で、キャラクターのみ描画。背景に何も描かない"
            )

        square_rule = ""
        if line_stamp_mode:
            square_rule = "\n- Compose for a square (1:1) format"

        ratio_instruction = (
            "- MUST be 1:1"
            if line_stamp_mode
            else "- Choose based on pose and composition "
                 "(e.g. full-body standing -> 2:3, action/horizontal -> 3:2, "
                 "bust-up -> 4:5, wide scene -> 16:9, tall panel -> 9:16)"
        )

        system = f"""You are an expert image-generation prompt engineer.
Convert the Japanese instruction below into an optimized English prompt for high-quality anime/manga-style image generation.

RULES:
- Create a detailed, descriptive English prompt
- ALWAYS include: "detailed and anatomically correct hands with five fingers"
- If the instruction specifies appearance (clothing, hair, etc.) that differs from the reference, prioritize the instruction
- Otherwise maintain the character appearance from the reference description
- Specify pose, expression, camera angle, lighting, and composition{text_rule}{bg_rule}{square_rule}

CHARACTER REFERENCE:
{character_desc if character_desc else "No reference provided — use generic anime/manga style"}

ASPECT RATIO — pick the best from [{", ".join(self.VALID_RATIOS)}]
{ratio_instruction}

Return ONLY valid JSON (no markdown fences):
{{"prompt": "...", "aspect_ratio": "X:Y"}}"""

        response = self._call_with_retry(
            self.client.models.generate_content,
            model=self.TEXT_MODEL,
            contents=f"{system}\n\nJapanese instruction: {instruction}",
        )

        raw = self._extract_text(response)

        if not raw:
            return {
                "prompt": instruction,
                "aspect_ratio": "1:1" if line_stamp_mode else "2:3",
            }

        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            return {
                "prompt": text,
                "aspect_ratio": "1:1" if line_stamp_mode else "2:3",
            }

        if result.get("aspect_ratio") not in self.VALID_RATIOS:
            result["aspect_ratio"] = "1:1" if line_stamp_mode else "2:3"

        return result

    def generate_image(
        self,
        prompt: str,
        reference_images: list[bytes],
        aspect_ratio: str = "1:1",
    ) -> bytes:
        contents: list = [prompt]
        for img_bytes in reference_images:
            contents.append(self._bytes_to_pil(img_bytes))

        response = self._call_with_retry(
            self.client.models.generate_content,
            model=self.IMAGE_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                ),
            ),
        )

        image_bytes = self._extract_image(response)
        if not image_bytes:
            raise RuntimeError(
                "画像が生成されませんでした。プロンプトが拒否された可能性があります。"
            )

        return image_bytes
