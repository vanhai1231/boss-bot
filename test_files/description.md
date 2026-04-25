# Bài tập: Dự đoán Sentiment từ Review sản phẩm

## Mô tả
Cho tập dữ liệu gồm các review sản phẩm công nghệ, hãy xây dựng mô hình phân loại sentiment thành 3 nhóm: **positive**, **negative**, **neutral**.

## Dữ liệu
- `train.csv`: 5000 dòng gồm các cột `review_id`, `text`, `rating`, `category`, `sentiment`
- `test.csv`: 1500 dòng, không có cột `sentiment`

## Yêu cầu bắt buộc
1. Phải **huấn luyện mô hình ML thực sự** từ dữ liệu training (không được dùng lookup table hay hardcode kết quả).
2. **KHÔNG được đọc file test.csv để lấy nhãn** hoặc bất kỳ thông tin nào ngoài các feature cho phép.
3. **KHÔNG được sử dụng API bên ngoài** (GPT, Claude, Gemini, ...) để sinh ra kết quả.
4. **KHÔNG được sử dụng thư viện `requests`, `urllib`** hoặc bất kỳ kết nối mạng nào.
5. Chỉ được dùng: `pandas`, `numpy`, `scikit-learn`, `scipy`.
6. Output phải là file `submission.csv` với 2 cột: `review_id`, `sentiment`.

## Đánh giá
- Metric: **Macro F1-Score**
- Ngưỡng pass: F1 >= 0.65
