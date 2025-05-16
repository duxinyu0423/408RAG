

import os

current_path = os.getcwd()
print(f"当前工作路径：{current_path}")

custom_dir = "/home/user/lwh/dxyu/llm-universe-test/UserTMP"
if not os.path.exists(custom_dir):
    print('False')
else:
    print('True')



from tempfile import TemporaryDirectory
import os
tmp_dir = None
with TemporaryDirectory(
    suffix="_DB",  # 自定义后缀（可选）
    prefix="project_",  # 自定义前缀（可选）
    dir="/home/user/lwh/dxyu/llm-universe-test/UserTMP"  # 指定存储路径（默认使用系统临时目录）
) as tmp_dir:
    if not os.path.exists(tmp_dir):
        print('False')
    else:
        print('True')
    print(f"临时目录已创建：{tmp_dir}")
    
    # 在临时目录中创建文件
    tmp_file = os.path.join(tmp_dir, "test.txt")
    with open(tmp_file, "w") as f:
        f.write("临时数据示例")
        
    # 程序运行期间目录持续存在
# 退出with代码块后自动删除目录

if not os.path.exists(tmp_dir):
    print('False')
else:
    print('True')