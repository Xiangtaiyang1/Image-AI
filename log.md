# This is the log file
# 这是记录文件<br>
## Table of Content/目录:
[2026/09/11](#20260911)
[2026/09/12](#20260912)
### Format/格式:
Special Note/注: Need to add the date to the table of content/需要将日期添加到目录
Table of Content/目录:
```markdown
[yyyy/mm/dd](yyyymmdd)
```
Log/日志:
```markdown
###### yyyy/mm/dd:<br>
Content/内容
```
### Example/例子:
###### 2000/01/01:<br>
The beginning of the 21th century<br>
21世纪的开始
<br><br>
### **LOG/日志:**
###### 2026/09/11
~~911事件25周年纪念日~~<br>
将项目开源到 `github`<br>
使用 MIT 协议，更新 README，上传文件
###### 2026/09/12
~~终于不是 9.11 了~~<br>
1. 发现了 AI 的第一个bug：
    识别图像后因**映射问题**导致结果均分布于左上角<br>
    相关文件和代码:<br>
    `predict.py`:<br>
    ```python
    def process_image(model, image_path, output_dir, args, classes=None):
        ...
        if args.save_vis:
            ...
            if img is not None:
                vis_img = draw_detections(img, detections, classes)
                ...
    
    return detections
    ```
    这里调用了draw_detections函数
    ```python
    def draw_detections(image, detections, classes=None):
        ...
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            ...
        return vis_img
    ```
    这里的映射存在巨大问题。bbox映射方式为[x, y, w, h]，然而这里却使用了图像框的四个角的坐标\(`[x1, y1, x2, y2]`\)。同时，AI 进行处理时使用的是`640x640`大小的图片，输出是却没有依照原图像比例进行映射。
