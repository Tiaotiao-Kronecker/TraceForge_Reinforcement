#!/usr/bin/env python3
"""
从已运行的 viser 3D 可视化页面（如 localhost:8081）截取 25 帧并合成 GIF。

使用前请确保：
1. 可视化已启动并在浏览器中打开（如 http://localhost:8081）
2. 安装依赖: pip install playwright imageio pillow && playwright install chromium

用法:
    # 默认截取 25 帧、输出 capture_8081.gif
    python scripts/visualization/capture_viser_to_gif.py --port 8081

    # 指定帧数和输出路径
    python scripts/visualization/capture_viser_to_gif.py --port 8081 --num_frames 25 --out capture.gif
"""

import argparse
import io
import time
import os

def main():
    parser = argparse.ArgumentParser(description="截取 viser 页面 25 帧并导出为 GIF")
    parser.add_argument("--port", type=int, default=8081, help="viser 服务端口")
    parser.add_argument("--num_frames", type=int, default=25, help="截取帧数")
    parser.add_argument("--out", type=str, default=None, help="输出 GIF 路径，默认 capture_<port>.gif")
    parser.add_argument("--delay_per_frame_ms", type=int, default=200, help="每帧截取后等待毫秒（等渲染稳定）")
    parser.add_argument("--gif_duration_ms", type=int, default=100, help="GIF 每帧显示时长（毫秒）")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("请先安装: pip install playwright && playwright install chromium")
        return 1

    try:
        import imageio
    except ImportError:
        print("请先安装: pip install imageio")
        return 1

    url = f"http://localhost:{args.port}"
    out_path = args.out or os.path.join(os.getcwd(), f"capture_{args.port}.gif")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    delay_sec = args.delay_per_frame_ms / 1000.0
    frames = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 720})
        page = context.new_page()
        page.goto(url, wait_until="networkidle", timeout=15000)
        time.sleep(2)

        max_slider = max(1, args.num_frames - 1)
        # 查找第一个 range 滑块（viser 里“时间步”通常是第一个）
        slider = page.locator('input[type="range"]').first
        try:
            slider.wait_for(state="visible", timeout=5000)
            has_slider = True
        except Exception:
            has_slider = False
            print("未找到时间滑块，将截取当前画面重复多帧。")

        for t in range(args.num_frames):
            if has_slider:
                try:
                    # 设置 range 的值并触发 input（min/max 由页面决定）
                    slider.evaluate(
                        f"""el => {{
                            const v = Math.min({t}, {max_slider});
                            el.value = v;
                            el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        }}"""
                    )
                except Exception:
                    pass
            time.sleep(delay_sec)
            screenshot = page.screenshot(type="png")
            frames.append(imageio.imread(io.BytesIO(screenshot)))

        browser.close()

    duration_sec = args.gif_duration_ms / 1000.0
    imageio.mimsave(out_path, frames, duration=duration_sec, loop=0)
    print(f"已保存: {out_path} ({len(frames)} 帧)")
    return 0


if __name__ == "__main__":
    exit(main())
