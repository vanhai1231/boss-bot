#!/bin/bash
# Cập nhật task và push lên Railway — chạy 1 lần là xong!

cd /Users/havanhai/shipd/bot

echo "🔍 Đang quét task..."
python sync_tasks.py

echo ""
echo "📦 Đang push lên GitHub..."
git add -A
git commit -m "🔄 update tasks $(date '+%Y-%m-%d %H:%M')"
git push

echo ""
echo "✅ Xong! Railway sẽ tự deploy trong 1-2 phút."
