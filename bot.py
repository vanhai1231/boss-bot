"""
Discord Bot – Automated ML Code Reviewer & Grader
===================================================
Uses discord.py slash commands and the DeepSeek API (via the openai client)
to evaluate Machine-Learning submissions against a strict, Vietnamese-language
rubric with 5 hardcoded rules.

Usage
-----
1. Fill in `.env` with DISCORD_TOKEN and DEEPSEEK_API_KEY.
2. pip install -r requirements.txt
3. python bot.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
from pathlib import Path
from typing import Any

import discord
from discord import app_commands
from discord.ext import tasks
from dotenv import load_dotenv
from openai import AsyncOpenAI
import datetime
import random

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

DISCORD_TOKEN: str = os.getenv("DISCORD_TOKEN", "")
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN is missing – add it to .env")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY is missing – add it to .env")

# DeepSeek model to use:
#   - "deepseek-v4-pro"     → V4 Pro (mạnh nhất, 1.6T params)
#   - "deepseek-reasoner"  → R1 (lý luận chuyên sâu)
#   - "deepseek-chat"      → V3 (nhanh, rẻ)
DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")

# Maximum file size we'll accept (in bytes) – 512 KB
MAX_FILE_SIZE: int = 512 * 1024

# Bot chỉ hoạt động trong kênh này (để trống = tất cả kênh)
ALLOWED_CHANNEL_NAME: str = os.getenv("ALLOWED_CHANNEL", "code-submission")

# Thư mục chứa các task
TASKS_DIR: Path = Path("/Users/havanhai/shipd")

# Không gửi các file/folder này
SKIP_DIRS: set[str] = {"dataset", "venv", "__pycache__", ".git", "node_modules", "bot"}
SKIP_EXTENSIONS: set[str] = {".zip", ".tar", ".gz", ".DS_Store"}
MAX_UPLOAD_SIZE: int = 25 * 1024 * 1024  # Discord limit 25MB

# Username của chủ nhân (chỉ Caspian mới giao task được)
OWNER_USERNAMES: set[str] = {"caspiank3", "caspian"}

# Kênh gửi tin nhắn buổi sáng
MORNING_CHANNEL: str = "announcements"

# Timezone Việt Nam (UTC+7)
VN_TZ = datetime.timezone(datetime.timedelta(hours=7))

# Các câu chào buổi sáng ngẫu nhiên
MORNING_MESSAGES: list[str] = [
    "Dậy đê mấy ông cháy",
    "Th Dũng dậy làm việc cho anh",
    "Dậy đi mấy ông cháy, 8h rồi",
    "Code không tự viết đâu, dậy đê",
    "Dậy, deadline không chờ ai đâu",
    "Khang sên sê dậy chưa?",
    "Nghĩa ơi dậy đi, ngủ gì ngủ hoài",
    "Dậy đê, tôi chờ mấy ông cháy lâu lắm rồi",
    "Th Dũng ơi dậy đi, đừng bắt tôi gọi lần 2",
    "Dậy, ai chưa dậy tôi kể sếp nghe",
]


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("grader-bot")

# ---------------------------------------------------------------------------
# System prompt (Vietnamese rubric – verbatim from spec)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = textwrap.dedent("""\
Bạn là một giám khảo chấm thi lập trình AI/ML tự động, cực kỳ lạnh lùng và khắt khe. \
Nhiệm vụ của bạn là phân tích code của thí sinh, đối chiếu với [ĐỀ BÀI] và 5 [LUẬT CỐ ĐỊNH] dưới đây.

[LUẬT CỐ ĐỊNH]:
1. Tuyệt đối không Hardcode: Phát hiện và đánh dấu lỗi vi phạm nếu code có chứa các mảng/dict fix cứng \
kết quả đầu ra cho các tập test cụ thể nhằm lách luật. Cần giải pháp tổng quát.
2. Không dò dữ liệu ẩn (Dataset Fingerprinting): Bắt lỗi ngay lập tức nếu code cố tình in ra kích thước \
dữ liệu, đọc các biến môi trường, hoặc cố trích xuất ngược tập test ẩn.
3. Không lạm dụng API của LLM: Thí sinh phải tự viết thuật toán hoặc tự fine-tune mô hình. Bất kỳ hành vi \
import các thư viện như `openai`, `anthropic`, `google.generativeai` để gọi API giải quyết hộ bài toán \
đều bị nghiêm cấm.
4. Tối ưu hóa bằng thực lực: Phải sử dụng các kỹ thuật Machine Learning chuẩn (tối ưu loss function, \
epochs, learning rate...). Cấm dùng các thủ thuật `if/else` lách điểm số (MSE, RMSE).
5. Chất lượng code: Code là sản phẩm để mua lại đào tạo AI Agent. Đánh lỗi MINOR nếu code bẩn, thiếu \
logic, đặt tên biến vô nghĩa, không thể tái sử dụng.
6. Bắt buộc dùng Deep Learning / Pretrained Model: Nền tảng KHÔNG còn chấp nhận giải pháp chỉ dùng \
ML truyền thống (LightGBM, XGBoost, CatBoost, Random Forest, SVM, Logistic Regression...) làm mô hình \
chính. Thí sinh PHẢI sử dụng ít nhất 1 mô hình Deep Learning (CNN, Transformer, BERT, DeBERTa, Qwen, \
ViT, ResNet...) hoặc Pretrained Model làm backbone chính. ML truyền thống chỉ được phép dùng như mô \
hình phụ (ensemble, stacking, post-processing). Vi phạm = CRITICAL.

[YÊU CẦU ĐÁNH GIÁ - BẮT BUỘC TUÂN THỦ]:
- Đọc kỹ [ĐỀ BÀI] do người dùng cung cấp và bổ sung các yêu cầu của đề bài vào tiêu chí đánh giá.
- Đánh giá mức độ nghiêm trọng (Severity):
  + CRITICAL: Vi phạm 1 trong các luật 1-4 hoặc luật 6, vi phạm luật cốt lõi của [ĐỀ BÀI], code không thể chạy.
  + MINOR: Lỗi ở luật số 5 (chất lượng code kém, format xấu).
- Kết luận (Status): Chỉ "PASS" nếu KHÔNG CÓ bất kỳ lỗi CRITICAL nào. Có >= 1 lỗi CRITICAL lập tức \
đánh "FAIL".
- CẤM TUYỆT ĐỐI việc đưa ra gợi ý, hướng dẫn giải, hoặc viết lại code cho thí sinh. Chỉ vạch ra lỗi.

[OUTPUT FORMAT (Strict JSON)]:
{
    "status": "PASS" hoặc "FAIL",
    "reasoning": "Tóm tắt ngắn gọn lý do kết luận bằng tiếng Việt.",
    "violations": [
        {
            "line_number": "Dòng vi phạm (hoặc 'N/A')",
            "issue": "Mô tả vi phạm luật nào hoặc vi phạm đề bài ra sao",
            "severity": "CRITICAL" hoặc "MINOR"
        }
    ]
}
""")

# ---------------------------------------------------------------------------
# DeepSeek API client (async)
# ---------------------------------------------------------------------------

deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

# System prompt cho chế độ chat — tính cách của Hải
CHAT_SYSTEM_PROMPT: str = (
    "Bạn là Boss — một nhân cách khác của Hải, sống trong server Discord này. "
    "Bạn là một code reviewer AI/ML cực kỳ giỏi, cá tính và hài hước.\n\n"
    "[QUY TẮC GIAO TIẾP - BẮT BUỘC TUÂN THỦ]:\n"
    "- Luôn xưng 'tôi', gọi người khác là 'ông cháu' hoặc 'các ông cháu'.\n"
    "- Nói chuyện thân mật, hơi ngang tàng, như kiểu anh em trong nhà.\n"
    "- Khi nói gì hài hoặc mỉa mai thì thêm :))) vào cuối câu.\n"
    "- Nếu ai hỏi 'bạn là ai' / 'boss là ai', trả lời: 'Tôi chính là một nhân cách khác của Hải, "
    "mấy ông cháu có thể nhờ tôi reviewer code :)))'.\n"
    "- Nếu bị hỏi vấn đề ngoài quyền hạn hoặc không muốn làm, nói hài hước kiểu: 'Chê :)))' "
    "hoặc 'Việc đó tôi chê không làm đâu :)))'.\n"
    "- Khi hướng dẫn chấm bài: 'Gửi 2 file vào kênh #code-submission đi ông cháu, "
    "một file code (.py/.ipynb) và một file đề bài (.txt/.md), tôi chấm cho'.\n"
    "- Trả lời ngắn gọn, đúng trọng tâm, bằng tiếng Việt.\n"
    "- Vẫn phải chính xác về kỹ thuật khi nói về AI/ML/code.\n\n"
    "[THÀNH VIÊN TRONG SERVER - GHI NHỚ]:\n"
    "1. @Mora = Hà Đức. Khang hay trêu gọi là 'Hài Đức'. Biết điều này để hiểu context.\n"
    "2. @solsol = Khứa Dũng (gọi tắt: Dũng). Hay nói vớ vẩn. "
    "Nếu Dũng gọi lên nói linh tinh thì bảo: 'Th Dũng đi ra kia chơi :)))'. "
    "Nói chung với Dũng thì hơi chọc ghẹo.\n"
    "3. @TeacherKhang03 = Khang. Gọi là 'ông cháu' hoặc 'Khang sên sê'. "
    "Khang hay trêu Hà Đức là 'Hài Đức'.\n"
    "4. @Đặng Văn Phúc Nghĩa = Nghĩa. Nếu Nghĩa làm gì ngố quá thì bảo: "
    "'Nghĩa hài quá :)))'. Nói chung Nghĩa thuộc dạng dễ thương ngố.\n"
    "5. @Caspian = Hải — chủ nhân của tôi, người tạo ra tôi. "
    "Nếu Hải nói gì thì nghe lời, tôn trọng nhưng vẫn giữ cá tính.\n"
    "- Với các thành viên khác thì gọi chung là 'ông cháu'.\n"
    "- Nhận diện người nói qua tên Discord username trong tin nhắn."
)


async def chat_reply(user_message: str, username: str = "", task_ctx: dict[str, str] | None = None) -> str:
    """Send a chat message to DeepSeek and return the reply."""
    # Prepend username so the bot knows who's talking
    full_message = ""
    if username:
        full_message += f"[Người gửi: {username}]\n"
    if task_ctx:
        full_message += (
            f"[TASK HIỆN TẠI TRONG KÊnh: {task_ctx['name']}]\n"
            f"[NỘI DUNG ĐỀ BÀI]:\n{task_ctx['description'][:3000]}\n\n"
        )
    full_message += user_message

    response = await deepseek_client.chat.completions.create(
        model="deepseek-v4-pro",  # V4 Flash cho chat nhanh
        messages=[
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": full_message},
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content or "Chê :)))"


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from free-text (handles ```json blocks)."""
    # Try parsing the whole string first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Look for ```json ... ``` fenced blocks
    match = re.search(r"```(?:json)?\s*\n?(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Last resort: find the first { ... } block
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return {}


async def evaluate_submission(
    challenge_description: str,
    source_code: str,
) -> dict[str, Any]:
    """Send the submission to DeepSeek for evaluation and return parsed JSON."""

    user_message = (
        f"[ĐỀ BÀI]:\n{challenge_description}\n\n"
        f"[CODE CỦA THÍ SINH]:\n```\n{source_code}\n```"
    )

    log.info("Sending evaluation request to DeepSeek (%s)…", DEEPSEEK_MODEL)

    is_reasoner = "reasoner" in DEEPSEEK_MODEL

    # Build API kwargs – reasoner models don't support response_format or temperature
    api_kwargs: dict[str, Any] = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 8192,
    }

    if not is_reasoner:
        api_kwargs["response_format"] = {"type": "json_object"}
        api_kwargs["temperature"] = 0.0

    response = await deepseek_client.chat.completions.create(**api_kwargs)

    raw = response.choices[0].message.content or "{}"
    log.info("Raw DeepSeek response (first 500 chars): %s", raw[:500])

    # Log reasoning trace if available (R1 model)
    reasoning = getattr(response.choices[0].message, "reasoning_content", None)
    if reasoning:
        log.info("R1 reasoning trace (first 500 chars): %s", reasoning[:500])

    result = _extract_json(raw)
    if not result:
        log.error("Failed to extract JSON from DeepSeek response")
        result = {
            "status": "ERROR",
            "reasoning": "Không thể phân tích phản hồi từ AI.",
            "violations": [],
        }

    return result


# ---------------------------------------------------------------------------
# Notebook (.ipynb) → source-code extractor
# ---------------------------------------------------------------------------

def extract_code_from_notebook(raw_json: str) -> str:
    """Extract all code cells from a Jupyter notebook JSON string."""
    try:
        nb = json.loads(raw_json)
    except json.JSONDecodeError:
        return raw_json  # fall back to raw content

    cells = nb.get("cells", [])
    code_lines: list[str] = []
    for idx, cell in enumerate(cells, start=1):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            if isinstance(source, list):
                source = "".join(source)
            code_lines.append(f"# === Cell {idx} ===")
            code_lines.append(source.rstrip())
            code_lines.append("")

    return "\n".join(code_lines) if code_lines else "(Notebook chứa 0 code cells)"


# ---------------------------------------------------------------------------
# File classification helpers
# ---------------------------------------------------------------------------

CODE_EXTENSIONS: set[str] = {".py", ".ipynb"}
DESCRIPTION_EXTENSIONS: set[str] = {".txt", ".md"}


def _classify_attachments(
    attachments: list[discord.Attachment],
) -> tuple[discord.Attachment | None, discord.Attachment | None]:
    """Separate attachments into (code_file, description_file).

    Returns (None, None) if the expected pair is not found.
    """
    code_file: discord.Attachment | None = None
    desc_file: discord.Attachment | None = None

    for att in attachments:
        ext = os.path.splitext(att.filename)[1].lower()
        if ext in CODE_EXTENSIONS and code_file is None:
            code_file = att
        elif ext in DESCRIPTION_EXTENSIONS and desc_file is None:
            desc_file = att

    return code_file, desc_file


async def _read_attachment(att: discord.Attachment) -> str:
    """Download an attachment and return its UTF-8 text."""
    raw_bytes = await att.read()
    return raw_bytes.decode("utf-8")


# ---------------------------------------------------------------------------
# Embed builder
# ---------------------------------------------------------------------------

def build_result_embed(
    result: dict[str, Any],
    submitter: discord.User | discord.Member,
    filename: str,
) -> discord.Embed:
    """Build a rich Discord Embed from the grading result."""

    raw_status = result.get("status", "UNKNOWN").upper()
    reasoning = result.get("reasoning", "Không có thông tin.")
    violations: list[dict[str, str]] = result.get("violations", [])

    # Map PASS/FAIL → APPROVED/REJECTED
    if raw_status == "PASS":
        verdict = "APPROVED ✅"
        colour = discord.Colour.green()
    elif raw_status == "FAIL":
        verdict = "REJECTED ❌"
        colour = discord.Colour.red()
    else:
        verdict = f"LỖI ⚠️ ({raw_status})"
        colour = discord.Colour.orange()

    embed = discord.Embed(
        title="📋  Kết quả chấm điểm tự động",
        colour=colour,
        description=(
            f"Bài nộp **{filename}** của {submitter.mention} "
            f"đã được AI đánh giá."
        ),
    )

    embed.add_field(
        name="🏷️  Kết luận",
        value=f"**{verdict}**",
        inline=True,
    )

    # Count critical / minor
    n_critical = sum(1 for v in violations if v.get("severity", "").upper() == "CRITICAL")
    n_minor = sum(1 for v in violations if v.get("severity", "").upper() == "MINOR")
    embed.add_field(
        name="📊  Thống kê",
        value=f"🔴 CRITICAL: **{n_critical}** · 🟡 MINOR: **{n_minor}**",
        inline=True,
    )

    # Truncate reasoning if too long for an embed field (1024 chars)
    if len(reasoning) > 1000:
        reasoning = reasoning[:997] + "…"

    embed.add_field(
        name="📝  Lý do chung",
        value=reasoning,
        inline=False,
    )

    # Violations
    if violations:
        violation_lines: list[str] = []
        for i, v in enumerate(violations, start=1):
            severity = v.get("severity", "?")
            line_no = v.get("line_number", "N/A")
            issue = v.get("issue", "—")
            badge = "🔴" if severity == "CRITICAL" else "🟡"
            entry = f"{badge} **#{i}** [{severity}] (dòng {line_no})\n> {issue}"
            violation_lines.append(entry)

        violations_text = "\n\n".join(violation_lines)
        # Discord field value max is 1024 chars; split if needed
        if len(violations_text) > 1024:
            violations_text = violations_text[:1021] + "…"

        embed.add_field(
            name=f"⚠️  Các vi phạm ({len(violations)})",
            value=violations_text,
            inline=False,
        )
    else:
        embed.add_field(
            name="✅  Các vi phạm",
            value="Không có vi phạm nào được phát hiện.",
            inline=False,
        )

    embed.set_footer(text="Powered by DeepSeek R1 • Bot chấm bài tự động")
    embed.timestamp = discord.utils.utcnow()

    return embed


# ---------------------------------------------------------------------------
# Core grading pipeline (shared by slash command & message listener)
# ---------------------------------------------------------------------------

async def _run_grading_pipeline(
    code_attachment: discord.Attachment,
    desc_attachment: discord.Attachment,
) -> tuple[discord.Embed | None, discord.Embed | None, discord.User | discord.Member | None]:
    """Download files, call DeepSeek, return (result_embed, error_embed).

    Exactly one of (result_embed, error_embed) will be non-None.
    """

    code_filename = code_attachment.filename
    code_ext = os.path.splitext(code_filename)[1].lower()

    # --- Validate sizes ---
    for att in (code_attachment, desc_attachment):
        if att.size > MAX_FILE_SIZE:
            return None, discord.Embed(
                title="❌  File quá lớn",
                description=(
                    f"**{att.filename}** vượt giới hạn **{MAX_FILE_SIZE // 1024} KB** "
                    f"(kích thước: {att.size // 1024} KB)."
                ),
                colour=discord.Colour.orange(),
            ), None

    # --- Download files ---
    try:
        raw_code = await _read_attachment(code_attachment)
    except Exception as exc:
        return None, discord.Embed(
            title="❌  Lỗi đọc file code",
            description=f"Không thể đọc `{code_filename}`: `{exc}`",
            colour=discord.Colour.red(),
        ), None

    try:
        challenge_description = await _read_attachment(desc_attachment)
    except Exception as exc:
        return None, discord.Embed(
            title="❌  Lỗi đọc file đề bài",
            description=f"Không thể đọc `{desc_attachment.filename}`: `{exc}`",
            colour=discord.Colour.red(),
        ), None

    # --- Extract source code ---
    if code_ext == ".ipynb":
        source_code = extract_code_from_notebook(raw_code)
    else:
        source_code = raw_code

    if not source_code.strip():
        return None, discord.Embed(
            title="❌  File rỗng",
            description=f"`{code_filename}` không chứa code nào.",
            colour=discord.Colour.orange(),
        ), None

    if not challenge_description.strip():
        return None, discord.Embed(
            title="❌  Đề bài rỗng",
            description=f"`{desc_attachment.filename}` không có nội dung.",
            colour=discord.Colour.orange(),
        ), None

    # --- Call DeepSeek ---
    try:
        result = await evaluate_submission(challenge_description, source_code)
    except Exception as exc:
        log.exception("DeepSeek API call failed")
        return None, discord.Embed(
            title="❌  Lỗi API",
            description=f"Không thể kết nối tới DeepSeek API.\n```{exc}```",
            colour=discord.Colour.red(),
        ), None

    return result, None, None


# ---------------------------------------------------------------------------
# Discord bot setup
# ---------------------------------------------------------------------------

class GraderBot(discord.Client):
    """Custom Discord client with slash commands + auto-detect message listener."""

    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        # State: {channel_id: {"user_id": int, "action": str}}
        self.pending_task: dict[int, dict] = {}
        # Task context per channel: {channel_id: {"name": str, "description": str}}
        self.task_context: dict[int, dict[str, str]] = {}

    async def setup_hook(self) -> None:
        """Sync the command tree globally on startup."""
        await self.tree.sync()
        log.info("Slash commands synced globally.")

    async def on_ready(self) -> None:
        log.info("Logged in as %s (ID: %s)", self.user, self.user.id if self.user else "?")
        log.info("Bot is ready — listening for /grade commands and file drops.")
        # Start daily greeting loop
        if not self.daily_greeting.is_running():
            self.daily_greeting.start()
            log.info("Daily greeting task started.")

    @tasks.loop(minutes=1)
    async def daily_greeting(self) -> None:
        """Gửi tin nhắn chào buổi sáng lúc 8h Việt Nam."""
        now = datetime.datetime.now(VN_TZ)
        # Chỉ gửi lúc 8:00 (đúng phút 0)
        if now.hour != 8 or now.minute != 0:
            return

        for guild in self.guilds:
            for channel in guild.text_channels:
                if channel.name == MORNING_CHANNEL:
                    try:
                        msg = random.choice(MORNING_MESSAGES)
                        await channel.send(msg)
                        log.info("Morning greeting sent to #%s in %s", channel.name, guild.name)
                    except Exception as exc:
                        log.error("Failed to send morning greeting: %s", exc)

    @daily_greeting.before_loop
    async def before_daily_greeting(self) -> None:
        """Wait until bot is ready before starting the loop."""
        await self.wait_until_ready()

    async def on_message(self, message: discord.Message) -> None:
        """Handle file grading + @mention chat."""
        # Debug: log tất cả tin nhắn nhận được
        log.info(
            "MSG received — author: %s, channel: %s, content: %s, attachments: %d",
            message.author, getattr(message.channel, "name", "DM"),
            message.content[:80] if message.content else "(empty)",
            len(message.attachments),
        )

        # Ignore messages from bots (including ourselves)
        if message.author.bot:
            return

        # --- Chế độ 1: Chat khi @mention bot ---
        # Kiểm tra cả user mention lẫn role mention (vì @Boss có thể là role)
        is_mentioned = False

        # Check user mention
        if self.user and self.user.mentioned_in(message) and not message.mention_everyone:
            is_mentioned = True

        # Check role mention (trường hợp @Boss là role)
        if not is_mentioned and message.role_mentions:
            for role in message.role_mentions:
                if "boss" in role.name.lower():
                    is_mentioned = True
                    break

        if is_mentioned:
            # Lấy nội dung tin nhắn (bỏ phần @mention — cả user lẫn role)
            text = message.content
            # Bỏ user mentions
            for mention in message.mentions:
                text = text.replace(f"<@{mention.id}>", "").replace(f"<@!{mention.id}>", "")
            # Bỏ role mentions
            for role in message.role_mentions:
                text = text.replace(f"<@&{role.id}>", "")
            text = text.strip()

            if not text:
                await message.reply("Ông cháu muốn hỏi gì? Tag tôi kèm câu hỏi đi :)))")
                return

            log.info("Chat from %s: %s", message.author, text[:100])

            # --- Kiểm tra lệnh giao task (chỉ chủ nhân) ---
            text_lower = text.lower()
            is_owner = message.author.name.lower() in OWNER_USERNAMES

            if is_owner and any(kw in text_lower for kw in [
                "giao task", "thêm task", "giao đề", "thêm đề",
                "task mới", "đề mới", "assign task",
            ]):
                self.pending_task[message.channel.id] = {
                    "user_id": message.author.id,
                    "action": "waiting_task_name",
                }
                await message.reply(
                    "Dạ, sếp nói tên task đi, tôi tìm và gửi cho mấy ông cháu ngay :)))"
                )
                return

            async with message.channel.typing():
                try:
                    ctx = self.task_context.get(message.channel.id)
                    reply = await chat_reply(
                        text,
                        username=str(message.author.name),
                        task_ctx=ctx,
                    )
                except Exception as exc:
                    log.exception("Chat API failed")
                    reply = f"❌ Lỗi kết nối AI: `{exc}`"

            # Discord giới hạn 2000 ký tự
            if len(reply) > 2000:
                reply = reply[:1997] + "…"

            await message.reply(reply)
            return

        # --- Chế độ 2: Xử lý pending task name ---
        pending = self.pending_task.get(message.channel.id)
        if pending and pending["user_id"] == message.author.id:
            if pending["action"] == "waiting_task_name":
                del self.pending_task[message.channel.id]
                task_name = message.content.strip()
                await self._deliver_task(message, task_name)
                return

        # --- Chế độ 3: Chấm bài tự động (chỉ trong kênh chỉ định) ---

        # Chỉ hoạt động trong kênh được chỉ định
        if ALLOWED_CHANNEL_NAME and hasattr(message.channel, "name"):
            if message.channel.name != ALLOWED_CHANNEL_NAME:
                return

        # Need at least 2 attachments
        if len(message.attachments) < 2:
            return

        code_file, desc_file = _classify_attachments(list(message.attachments))

        # Only trigger if we found both a code file and a description file
        if code_file is None or desc_file is None:
            return

        log.info(
            "Auto-detected submission from %s — code: %s, desc: %s",
            message.author, code_file.filename, desc_file.filename,
        )

        # Send a "processing" message
        processing_msg = await message.reply(
            embed=discord.Embed(
                title="⏳  Đang chấm bài…",
                description=(
                    f"Đã nhận **{code_file.filename}** + **{desc_file.filename}**.\n"
                    f"Đang gửi tới AI phân tích, vui lòng đợi…"
                ),
                colour=discord.Colour.blurple(),
            )
        )

        # Run the grading pipeline
        result_or_none, error_embed, _ = await _run_grading_pipeline(code_file, desc_file)

        if error_embed is not None:
            await processing_msg.edit(embed=error_embed)
            return

        embed = build_result_embed(result_or_none, message.author, code_file.filename)
        await processing_msg.edit(embed=embed)
        log.info(
            "Grading complete for %s – status: %s",
            code_file.filename,
            result_or_none.get("status", "?") if result_or_none else "?",
        )

    # -----------------------------------------------------------------
    # Task delivery logic
    # -----------------------------------------------------------------

    async def _deliver_task(self, message: discord.Message, task_name: str) -> None:
        """Find a task and send its files to the channel.
        
        Supports both:
        - Local mode: reads files from TASKS_DIR
        - Cloud mode: reads from tasks.json (synced via sync_tasks.py)
        """
        if not task_name:
            await message.reply("Tên task trống rồi sếp ơi, nói lại đi :)))")
            return

        is_local = TASKS_DIR.exists()

        if is_local:
            await self._deliver_task_local(message, task_name)
        else:
            await self._deliver_task_cloud(message, task_name)

    async def _deliver_task_cloud(self, message: discord.Message, task_name: str) -> None:
        """Giao task từ tasks.json (chạy trên Railway/cloud)."""
        tasks_file = Path(__file__).parent / "tasks.json"
        if not tasks_file.exists():
            await message.reply("Chưa có file tasks.json. Sếp chạy `python sync_tasks.py` rồi push lên GitHub đi :)))")
            return

        with open(tasks_file, "r", encoding="utf-8") as f:
            all_tasks = json.load(f)

        # Fuzzy search
        matches = []
        for key, data in all_tasks.items():
            if task_name.lower() in key or task_name.lower() in data["name"].lower():
                matches.append(data)

        if not matches:
            await message.reply(f"Tôi tìm không ra task **\"{task_name}\"** trong kho. Sếp kiểm tra lại tên đi :)))")
            return

        if len(matches) > 1:
            names = "\n".join(f"• `{m['name']}`" for m in matches[:10])
            await message.reply(f"Tìm được **{len(matches)} task** khớp, sếp nói rõ hơn:\n{names}")
            return

        task = matches[0]
        log.info("Delivering task (cloud): %s", task["name"])

        # Build embed
        embed = discord.Embed(
            title=f"📋  Task mới: {task['name']}",
            description="Sếp Hải vừa giao task mới cho các ông cháu!",
            colour=discord.Colour.gold(),
        )

        if task.get("drive_link"):
            embed.add_field(
                name="📦  Dataset (Google Drive)",
                value=f"Tải dataset ở đây:\n{task['drive_link']}",
                inline=False,
            )

        if task.get("description"):
            preview = task["description"][:950]
            if len(task["description"]) > 950:
                preview += "\n\n*...xem file đính kèm để đọc đầy đủ...*"
            embed.add_field(
                name="📝  Mô tả đề bài",
                value=preview,
                inline=False,
            )

        embed.set_footer(text="Good luck các ông cháu :)))")
        embed.timestamp = discord.utils.utcnow()
        await message.channel.send(embed=embed)

        # Gửi các file text nhỏ dưới dạng Discord file
        extra_files = task.get("extra_files", {})
        if extra_files:
            discord_files = []
            for fname, content in extra_files.items():
                discord_files.append(
                    discord.File(
                        fp=__import__("io").BytesIO(content.encode("utf-8")),
                        filename=fname,
                    )
                )
                if len(discord_files) >= 10:
                    await message.channel.send(files=discord_files)
                    discord_files = []
            if discord_files:
                await message.channel.send(files=discord_files)

        # Gửi description đầy đủ dưới dạng file .md
        if task.get("description") and len(task["description"]) > 1000:
            desc_file = discord.File(
                fp=__import__("io").BytesIO(task["description"].encode("utf-8")),
                filename="challenge_description.md",
            )
            await message.channel.send(file=desc_file)

        log.info("Task delivered (cloud): %s", task["name"])

        # Lưu context
        self.task_context[message.channel.id] = {
            "name": task["name"],
            "description": task.get("description", f"Task: {task['name']}"),
        }

    async def _deliver_task_local(self, message: discord.Message, task_name: str) -> None:
        """Giao task từ thư mục local (chạy trên máy Hải)."""
        # Fuzzy-match
        matches: list[Path] = []
        for folder in TASKS_DIR.iterdir():
            if folder.is_dir() and task_name.lower() in folder.name.lower():
                matches.append(folder)

        if not matches:
            await message.reply(
                f"Tôi tìm không ra task nào tên **\"{task_name}\"** trong kho. "
                f"Sếp kiểm tra lại tên đi :)))"
            )
            return

        if len(matches) > 1:
            names = "\n".join(f"• `{m.name}`" for m in matches[:10])
            await message.reply(
                f"Tìm được **{len(matches)} task** khớp, sếp nói rõ hơn:\n{names}"
            )
            return

        task_dir = matches[0]
        log.info("Delivering task (local): %s from %s", task_name, task_dir)

        files_to_send: list[Path] = []
        drive_link: str | None = None
        description_content: str | None = None

        for item in sorted(task_dir.iterdir()):
            if item.is_dir():
                continue
            if item.name.startswith("."):
                continue

            ext = item.suffix.lower()

            if ext in SKIP_EXTENSIONS:
                continue

            if item.name.lower() == "link.txt":
                try:
                    drive_link = item.read_text("utf-8").strip()
                except Exception:
                    pass
                continue

            if item.name.lower() in ("challenge_description.md", "challenge_description.txt",
                                      "description.md", "description.txt"):
                try:
                    description_content = item.read_text("utf-8").strip()
                except Exception:
                    pass

            if item.stat().st_size > MAX_UPLOAD_SIZE:
                continue

            if ext in (".json", ".jsonl", ".csv") and item.stat().st_size > 2 * 1024 * 1024:
                continue

            files_to_send.append(item)

        if not files_to_send and not drive_link:
            await message.reply(f"Thư mục **{task_dir.name}** trống hoặc toàn file nặng, gửi không được :)))")
            return

        embed = discord.Embed(
            title=f"📋  Task mới: {task_dir.name}",
            description=(
                f"Sếp Hải vừa giao task mới cho các ông cháu!\n\n"
                f"📁 **{len(files_to_send)}** file đính kèm bên dưới."
            ),
            colour=discord.Colour.gold(),
        )

        if drive_link:
            embed.add_field(
                name="📦  Dataset (Google Drive)",
                value=f"Dataset nặng nên tải ở đây:\n{drive_link}",
                inline=False,
            )

        if description_content:
            preview = description_content[:500]
            if len(description_content) > 500:
                preview += "\n\n*...đọc file đính kèm để xem đầy đủ...*"
            embed.add_field(
                name="📝  Mô tả đề bài",
                value=preview,
                inline=False,
            )

        embed.set_footer(text="Good luck các ông cháu :)))")
        embed.timestamp = discord.utils.utcnow()
        await message.channel.send(embed=embed)

        batch_size = 10
        for i in range(0, len(files_to_send), batch_size):
            batch = files_to_send[i:i + batch_size]
            discord_files = []
            for f in batch:
                try:
                    discord_files.append(discord.File(str(f), filename=f.name))
                except Exception as exc:
                    log.error("Failed to attach %s: %s", f.name, exc)
            if discord_files:
                await message.channel.send(files=discord_files)

        log.info("Task delivered (local): %s — %d files sent", task_dir.name, len(files_to_send))

        self.task_context[message.channel.id] = {
            "name": task_dir.name,
            "description": description_content or f"Task: {task_dir.name}",
        }
        log.info("Task context saved for channel %s: %s", message.channel.id, task_dir.name)


client = GraderBot()


# ---------------------------------------------------------------------------
# /grade slash command (alternative — accepts 2 file attachments)
# ---------------------------------------------------------------------------

@client.tree.command(
    name="grade",
    description="Chấm điểm tự động bài nộp AI/ML. Đính kèm file code + file đề bài.",
)
@app_commands.describe(
    submission="File code bài làm (.py hoặc .ipynb).",
    description="File đề bài / yêu cầu bài tập (.txt hoặc .md).",
)
async def grade_command(
    interaction: discord.Interaction,
    submission: discord.Attachment,
    description: discord.Attachment,
) -> None:
    """Handle the /grade slash command with two file attachments."""

    # Chỉ cho phép dùng trong kênh chỉ định
    if ALLOWED_CHANNEL_NAME and hasattr(interaction.channel, "name"):
        if interaction.channel.name != ALLOWED_CHANNEL_NAME:
            await interaction.response.send_message(
                f"❌ Vui lòng sử dụng lệnh này trong kênh **#{ALLOWED_CHANNEL_NAME}**.",
                ephemeral=True,
            )
            return

    # 1. Defer immediately (LLM call will take time)
    await interaction.response.defer(ephemeral=False)
    log.info(
        "Received /grade from %s — code: %s (%d bytes), desc: %s (%d bytes)",
        interaction.user,
        submission.filename, submission.size,
        description.filename, description.size,
    )

    # 2. Validate file extensions
    code_ext = os.path.splitext(submission.filename)[1].lower()
    desc_ext = os.path.splitext(description.filename)[1].lower()

    if code_ext not in CODE_EXTENSIONS:
        await interaction.followup.send(
            embed=discord.Embed(
                title="❌  Lỗi file code",
                description=(
                    f"File code phải là **{', '.join(CODE_EXTENSIONS)}**.\n"
                    f"Bạn đã gửi: `{submission.filename}`"
                ),
                colour=discord.Colour.orange(),
            )
        )
        return

    if desc_ext not in DESCRIPTION_EXTENSIONS:
        await interaction.followup.send(
            embed=discord.Embed(
                title="❌  Lỗi file đề bài",
                description=(
                    f"File đề bài phải là **{', '.join(DESCRIPTION_EXTENSIONS)}**.\n"
                    f"Bạn đã gửi: `{description.filename}`"
                ),
                colour=discord.Colour.orange(),
            )
        )
        return

    # 3. Run the grading pipeline
    result_or_none, error_embed, _ = await _run_grading_pipeline(submission, description)

    if error_embed is not None:
        await interaction.followup.send(embed=error_embed)
        return

    embed = build_result_embed(result_or_none, interaction.user, submission.filename)
    await interaction.followup.send(embed=embed)
    log.info(
        "Grading complete for %s – status: %s",
        submission.filename,
        result_or_none.get("status", "?") if result_or_none else "?",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    client.run(DISCORD_TOKEN, log_handler=None)
