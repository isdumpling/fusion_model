echo "--- 开始第一阶段训练 ---"
python main.py
echo "--- 第一阶段训练完成 ---"

echo "--- 开始第二阶段训练 (阈值 0.98 -> 0.70) ---"
python main.py --use_curriculum_learning --initial_confidence_threshold 0.98 --final_confidence_threshold 0.70
echo "--- 第二阶段训练完成 ---"

echo "--- 开始第三阶段训练 (阈值 0.95 -> 0.60) ---"
python main.py --use_curriculum_learning --initial_confidence_threshold 0.95 --final_confidence_threshold 0.60
echo "--- 第三阶段训练完成 ---"

echo "--- 开始第四阶段训练 (阈值 0.90 -> 0.60) ---"
python main.py --use_curriculum_learning --initial_confidence_threshold 0.90 --final_confidence_threshold 0.60
echo "--- 第四阶段训练完成 ---"

echo "--- 所有训练阶段已完成 ---"