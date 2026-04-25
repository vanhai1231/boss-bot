# 🤖 Bot Chấm Bài AI/ML Tự Động

Discord Bot sử dụng **DeepSeek R1** (model lý luận mạnh nhất) để chấm điểm tự động các bài nộp Machine Learning.

## Tính năng

- **Kéo thả file vào chat** — Bot tự nhận diện 2 file và chấm ngay
- **Slash command `/grade`** — Cách chính thống, đính kèm 2 file
- **5 luật chấm cố định**: Chống hardcode, chống dò dữ liệu, chống lạm dụng LLM API, yêu cầu ML thực thụ, chất lượng code
- **Hỗ trợ Jupyter Notebook**: Tự động trích xuất code cells từ `.ipynb`
- **Kết quả trực quan**: Embed Discord — **APPROVED ✅** hoặc **REJECTED ❌**

## Cài đặt

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Cấu hình .env
cp .env.example .env
# Sửa file .env: thêm DISCORD_TOKEN và DEEPSEEK_API_KEY

# 3. Chạy bot
python bot.py
```

## Cấu hình `.env`

| Biến | Mô tả |
|------|--------|
| `DISCORD_TOKEN` | Token của Discord Bot |
| `DEEPSEEK_API_KEY` | API key từ DeepSeek |
| `DEEPSEEK_MODEL` | (Tuỳ chọn) Mặc định: `deepseek-reasoner` (R1) |

## Cách sử dụng

### Cách 1: Kéo thả file (đơn giản nhất)
Gửi **2 file** vào bất kỳ kênh chat nào:
1. `solution.py` hoặc `solution.ipynb` — file code bài làm
2. `description.txt` hoặc `description.md` — file mô tả đề bài

Bot sẽ tự nhận diện và chấm ngay!

### Cách 2: Slash command
```
/grade submission:[file code] description:[file đề bài]
```

## Kết quả trả về

Bot trả embed gồm:
- 🏷️ **Kết luận**: APPROVED ✅ hoặc REJECTED ❌
- 📊 **Thống kê**: Số lỗi CRITICAL / MINOR
- 📝 **Lý do chung**: Giải thích tổng quát
- ⚠️ **Danh sách vi phạm**: Chi tiết từng lỗi (dòng, mô tả, mức độ)
