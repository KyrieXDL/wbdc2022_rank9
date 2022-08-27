echo '开始测试'
start=$(date +%s)
sh src/test_src/scripts/run_inference_clipvit_contras_stack.sh
end=$(date +%s)
take=$(( end - start ))
echo "推理: ${take} s"


# 100019872221
# sudo docker build -t tione-wxdsj.tencentcloudcr.com/team-100026276407/team-008:lastsubmit .
# sudo docker build -t tione-wxdsj.tencentcloudcr.com/team-100026276407/team-008:lastsubmit .
# sudo docker build -t tione-wxdsj.tencentcloudcr.com/team-100026276407/baseline_0805:v3.0 .
# sudo docker push tione-wxdsj.tencentcloudcr.com/team-100026276407/team-008:lastsubmit

