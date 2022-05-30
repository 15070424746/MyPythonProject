Real-ESRGAN 的目标是开发出**实用的图像/视频修复算法**。<br>
我们在 ESRGAN 的基础上使用纯合成的数据来进行训练，以使其能被应用于实际的图片修复的场景（顾名思义：Real-ESRGAN）。

---

其他推荐的项目：<br/>
:arrow_forward: [GFPGAN](https://github.com/TencentARC/GFPGAN): 实用的人脸复原算法 <br>
:arrow_forward: [BasicSR](https://github.com/xinntao/BasicSR): 开源的图像和视频工具箱<br>
:arrow_forward: [facexlib](https://github.com/xinntao/facexlib): 提供与人脸相关的工具箱<br>
:arrow_forward: [HandyView](https://github.com/xinntao/HandyView): 基于PyQt5的图片查看器，方便查看以及比较 <br>

---

<!--------------------------- Updates --------------------------->

<details>
<summary>🚩<b>更新</b></summary>

- ✅ 更新动漫视频的小模型 **RealESRGAN AnimeVideo-v3**. 更多信息在 [anime video models](docs/anime_video_model.md) 和 [comparisons](docs/anime_comparisons.md)中.
- ✅ 添加了针对动漫视频的小模型, 更多信息在 [anime video models](docs/anime_video_model.md) 中.
- ✅ 添加了ncnn 实现：[Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan).
- ✅ 添加了 [*RealESRGAN_x4plus_anime_6B.pth*](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth)，对二次元图片进行了优化，并减少了model的大小。详情 以及 与[waifu2x](https://github.com/nihui/waifu2x-ncnn-vulkan)的对比请查看[**anime_model.md**](docs/anime_model.md)
- ✅支持用户在自己的数据上进行微调 (finetune)：[详情](docs/Training.md#Finetune-Real-ESRGAN-on-your-own-dataset)
- ✅ 支持使用[GFPGAN](https://github.com/TencentARC/GFPGAN)**增强人脸**
- ✅ 通过[Gradio](https://github.com/gradio-app/gradio)添加到了[Huggingface Spaces](https://huggingface.co/spaces)（一个机器学习应用的在线平台）：[Gradio在线版](https://huggingface.co/spaces/akhaliq/Real-ESRGAN)。感谢[@AK391](https://github.com/AK391)
- ✅ 支持任意比例的缩放：`--outscale`（实际上使用`LANCZOS4`来更进一步调整输出图像的尺寸）。添加了*RealESRGAN_x2plus.pth*模型
- ✅ [推断脚本](inference_realesrgan.py)支持: 1) 分块处理**tile**; 2) 带**alpha通道**的图像; 3) **灰色**图像; 4) **16-bit**图像.
- ✅ 训练代码已经发布，具体做法可查看：[Training.md](docs/Training.md)。

<!----------------- Projects that use RealESRGAN ------------------>

<details>
<summary>🧩<b>使用Real-ESRGAN的项目</b></summary>

&nbsp;&nbsp;&nbsp;&nbsp;👋 如果你开发/使用/集成了Real-ESRGAN, 欢迎联系我添加

- NCNN-Android: [RealSR-NCNN-Android](https://github.com/tumuyan/RealSR-NCNN-Android) by [tumuyan](https://github.com/tumuyan)
- VapourSynth: [vs-realesrgan](https://github.com/HolyWu/vs-realesrgan) by [HolyWu](https://github.com/HolyWu)
- NCNN: [Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)

&nbsp;&nbsp;&nbsp;&nbsp;**易用的图形界面**

- [Waifu2x-Extension-GUI](https://github.com/AaronFeng753/Waifu2x-Extension-GUI) by [AaronFeng753](https://github.com/AaronFeng753)
- [Squirrel-RIFE](https://github.com/Justin62628/Squirrel-RIFE) by [Justin62628](https://github.com/Justin62628)
- [Real-GUI](https://github.com/scifx/Real-GUI) by [scifx](https://github.com/scifx)
- [Real-ESRGAN_GUI](https://github.com/net2cn/Real-ESRGAN_GUI) by [net2cn](https://github.com/net2cn)
- [Real-ESRGAN-EGUI](https://github.com/WGzeyu/Real-ESRGAN-EGUI) by [WGzeyu](https://github.com/WGzeyu)
- [anime_upscaler](https://github.com/shangar21/anime_upscaler) by [shangar21](https://github.com/shangar21)
- [RealESRGAN-GUI](https://github.com/Baiyuetribe/paper2gui/blob/main/Video%20Super%20Resolution/RealESRGAN-GUI.md) by [Baiyuetribe](https://github.com/Baiyuetribe)

<details>
<summary>👀<b>Demo视频（B站）</b></summary>

- [大闹天宫片段](https://www.bilibili.com/video/BV1ja41117zb)



### :book: Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data

> [[论文](https://arxiv.org/abs/2107.10833)] &emsp; [项目主页] &emsp; [[YouTube 视频](https://www.youtube.com/watch?v=fxHWoDSSvSc)] &emsp; [[B站视频](https://www.bilibili.com/video/BV1H34y1m7sS/)] &emsp; [[Poster](https://xinntao.github.io/projects/RealESRGAN_src/RealESRGAN_poster.pdf)] &emsp; [[PPT](https://docs.google.com/presentation/d/1QtW6Iy8rm8rGLsJ0Ldti6kP-7Qyzy6XL/edit?usp=sharing&ouid=109799856763657548160&rtpof=true&sd=true)]<br>
> [Xintao Wang](https://xinntao.github.io/), Liangbin Xie, [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en) <br>
> Tencent ARC Lab; Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences


---

我们提供了一套训练好的模型（*RealESRGAN_x4plus.pth*)，可以进行4倍的超分辨率。<br>
**现在的 Real-ESRGAN 还是有几率失败的，因为现实生活的降质过程比较复杂。**<br>
而且，本项目对**人脸以及文字之类**的效果还不是太好，Real-ESRGAN 将会被长期支持，我会在空闲的时间中持续维护更新。

---

## :zap: 快速上手

### 普通图片

下载我们训练好的模型: [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)

```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
```

推断!

```bash
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs --face_enhance
```

结果在`results`文件夹

### 动画图片

<p align="center">
  <img src="https://raw.githubusercontent.com/xinntao/public-figures/master/Real-ESRGAN/cmp_realesrgan_anime_1.png">
</p>
有关[waifu2x](https://github.com/nihui/waifu2x-ncnn-vulkan)的更多信息和对比在[**anime_model.md**](docs/anime_model.md)中。
```bash
# 下载模型
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P experiments/pretrained_models
# 推断
python inference_realesrgan.py -n RealESRGAN_x4plus_anime_6B -i inputs
```

结果在`results`文件夹

### Python 脚本的用法

1. 虽然你使用了 X4 模型，但是你可以 **输出任意尺寸比例的图片**，只要实用了 `outscale` 参数. 程序会进一步对模型的输出图像进行缩放。

```console
Usage: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile -o outfile [options]...

A common command: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile --outscale 3.5 --face_enhance

  -h                   show this help
  -i --input           Input image or folder. Default: inputs
  -o --output          Output folder. Default: results
  -n --model_name      Model name. Default: RealESRGAN_x4plus
  -s, --outscale       The final upsampling scale of the image. Default: 4
  --suffix             Suffix of the restored image. Default: out
  -t, --tile           Tile size, 0 for no tile during testing. Default: 0
  --face_enhance       Whether to use GFPGAN to enhance face. Default: False
  --fp32               Whether to use half precision during inference. Default: False
  --ext                Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto
```

## :european_castle: 模型库

请参见 [docs/model_zoo.md](docs/model_zoo.md)

## :computer: 训练，在你的数据上微调（Fine-tune）

这里有一份详细的指南：[Training.md](docs/Training.md).
