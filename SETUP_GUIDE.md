# 🛠️ Hướng dẫn setup Bot Discord từ A → Z

## Bước 1: Tạo Bot trên Discord Developer Portal

1. Truy cập: **https://discord.com/developers/applications**
2. Đăng nhập tài khoản Discord
3. Nhấn nút **"New Application"** (góc trên bên phải)
4. Đặt tên bot, ví dụ: `ML Code Grader` → nhấn **Create**

---

## Bước 2: Lấy Bot Token

1. Ở menu bên trái, chọn **"Bot"**
2. Nhấn **"Reset Token"** → **Copy** token hiện ra
3. Dán token vào file `.env`:
   ```
   DISCORD_TOKEN=paste_token_vào_đây
   DEEPSEEK_API_KEY=sk-d5d6f4b74e224fe5a128c1981c60b8b8
   ```

> ⚠️ **QUAN TRỌNG**: Không share token này cho ai. Nếu lộ, vào đây reset lại.

---

## Bước 3: Bật Intents (quyền đọc tin nhắn)

Vẫn ở trang **Bot**, cuộn xuống phần **Privileged Gateway Intents**, **bật ON** cả 3:

- [x] **PRESENCE INTENT**
- [x] **SERVER MEMBERS INTENT**  
- [x] **MESSAGE CONTENT INTENT** ← Bắt buộc! Bot cần đọc file đính kèm

Nhấn **Save Changes**.

---

## Bước 4: Tạo link mời Bot vào Server

1. Ở menu bên trái, chọn **"OAuth2"** → **"URL Generator"**
2. Mục **SCOPES**, tick:
   - [x] `bot`
   - [x] `applications.commands`
3. Mục **BOT PERMISSIONS**, tick:
   - [x] `Read Messages/View Channels`
   - [x] `Send Messages`
   - [x] `Send Messages in Threads`
   - [x] `Embed Links`
   - [x] `Attach Files`
   - [x] `Read Message History`
   
   *(Hoặc tick `Administrator` cho nhanh nếu server cá nhân)*

4. Copy **URL** ở cuối trang → **mở link đó trong trình duyệt**
5. Chọn server muốn thêm bot → **Authorize**

---

## Bước 5: Cài đặt và chạy Bot

```bash
cd /Users/havanhai/shipd/bot

# Cài thư viện
pip install -r requirements.txt

# Chạy bot
python bot.py
```

Nếu thấy log hiện:
```
Logged in as ML Code Grader#xxxx
Bot is ready — listening for /grade commands and file drops.
```
→ **Bot đã online!** 🎉

---

## Bước 6: Dùng thử

### Cách nhanh nhất — Kéo thả file:
1. Vào kênh chat bất kỳ trên server
2. Kéo thả **2 file** cùng lúc:
   - `solution.py` — code bài làm
   - `description.txt` — đề bài
3. Gửi → Bot tự động chấm!

### Cách 2 — Slash command:
1. Gõ `/grade` trong chat
2. Đính kèm file code vào ô **submission**
3. Đính kèm file đề bài vào ô **description**
4. Enter → Đợi kết quả

---

## ❓ FAQ — Lỗi thường gặp

| Lỗi | Cách sửa |
|-----|----------|
| Bot không phản hồi kéo thả file | Kiểm tra đã bật **MESSAGE CONTENT INTENT** ở Bước 3 chưa |
| `403 Forbidden` | Bot thiếu quyền — mời lại với đủ permissions ở Bước 4 |
| Slash command không hiện | Đợi 1-2 phút để Discord sync commands, hoặc restart bot |
| `DISCORD_TOKEN is missing` | Kiểm tra file `.env` đã điền token chưa |
| DeepSeek API lỗi | Kiểm tra `DEEPSEEK_API_KEY` và số dư tài khoản DeepSeek |
