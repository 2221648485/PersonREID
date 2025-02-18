from PIL import Image, ImageFont, ImageDraw
import numpy as np

# 绘制中文框
def draw_with_cn(img, font, box, label="", color=(128, 128, 128)):
    img = Image.fromarray(img)
    lw = max(round(sum(img.size) / 2 * 0.003), 2)
    draw = ImageDraw.Draw(img)
    size = max(round(sum(img.size) / 2 * 0.035), 12)
    font = ImageFont.truetype(str(font), size)
    p1 = (box[0], box[1])  # 边框左上角坐标
    draw.rectangle(box, width=lw, outline=color)
    if label:  # 绘制标签
        w, h = font.getbbox(label)[2:]
        outside = p1[1] - h >= 0  # 判断是否能写在框上
        draw.rectangle(
            (p1[0], p1[1] - h if outside else p1[1],
             p1[0] + w + 1, p1[1] + 1 if outside else p1[1] + h + 1),
            fill=color)
        draw.text((p1[0], p1[1] - h if outside else p1[1]), label, fill=color, font=font)
    img = np.asarray(img)
    return img
