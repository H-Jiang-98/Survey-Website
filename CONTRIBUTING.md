# 贡献指南：添加一篇论文

本项目从 `markdown_files/` 下的 Obsidian markdown 笔记自动生成网站。**你只需提交一个 `.md` 文件**，
`papers.json`、图片、部署全部由 CI 自动完成。

## TL;DR

> 开分支 → 加一个符合模板、**图片全为在线 URL** 的 `.md` → `python3 serve.py` 本地自检
> （确认没被剔除）→ 只提交那个 `.md` → 开 PR（CI 会校验图片合规）→ 合并到 `main` →
> 1–2 分钟后网页自动更新。

## 操作流程

### 1. 同步并开分支

```bash
git checkout main && git pull
git checkout -b add-paper-xxx
```

### 2. 在 `markdown_files/` 新增 `YourPaper.md`

文件名即论文 id。最省事的做法是**复制一篇现有在线图论文**（如 `markdown_files/Wh0.md`、`markdown_files/SONIC.md`）改写。
必须满足下列约定：

| 项 | 要求 | 不满足的后果 |
|---|---|---|
| frontmatter `title` | **必填**，否则不被识别为论文 | 整篇被忽略 |
| `year` / `venue` / `tags` | 驱动卡片徽章与筛选 | 对应筛选维度里缺这篇 |
| `## 一句话结论`（或 `## 一句话总结`）段落 | 卡片摘要来源 | 摘要显示「(无摘要)」 |
| `## 关键图表` 的第一张图 | 卡片封面缩略图 | 卡片无封面 |
| **所有图片为在线 URL** `![](https://…)` | 团队策略（见下） | **整篇被剔除；PR CI 失败** |
| 论文链接 | 速查卡片 / 元信息表 / 或正文出现的 arXiv 图片 URL（可自动推断 abs 链接） | 标题不可点击跳转 |

frontmatter 示例：

```yaml
---
title: "Your Paper Title"
method_name: "YourMethod"
authors: [First Author, Second Author]
year: 2026
venue: arXiv
tags: [humanoid, vision-language-action, ...]
image_source: online
---
```

### 3. 图片策略（重要）

**所有图片必须是在线 URL**，即 `![](https://arxiv.org/html/…/x1.png)` 这种形式。

- ❌ 不允许 Obsidian 本地嵌入 `![[fig1.png]]`
- ❌ 不允许指向仓库内相对路径 `![](assets/fig1.png)`

原因：在线图由 arXiv 等服务器托管，仓库保持轻量；本地图需要把图片字节也提交，体积大。
含本地图的论文会被构建**剔除**，并在 PR 的 CI 中**直接失败**。

### 4. 本地自检（强烈建议）

```bash
python3 serve.py        # 重建 + 起本地服务器并打开浏览器
```

- 终端若出现 `⚠ 已剔除 … 本地图: …` → 你的图不合规，**改成在线 URL 再提交**。
- 也可只跑图片校验：`python3 build.py --check markdown_files/YourPaper.md`（合规 exit 0，违规 exit 1）。
- 浏览器里确认：封面/摘要正常、点标题能跳转、点卡片弹窗里公式/图/表渲染正常。

### 5. 提交（只提交源文件）

```bash
git add markdown_files/YourPaper.md     # site/papers.json 已被 .gitignore 忽略，勿提交
git commit -m "Add YourPaper"
git push -u origin add-paper-xxx
```

> 不要提交 `site/papers.json`——它是构建产物，CI 会重建。

### 6. 开 PR → CI 校验 → 合并

- 开 Pull Request 后，**`Validate image policy` 工作流**只校验本次改动的 markdown：
  含本地图片则 CI 失败，必须改成在线 URL 才能合并。
- Review 通过后合并到 `main`。

### 7. 自动构建与上线

合并到 `main` 触发 `Build & Deploy` 工作流：`build.py` 重建 → 部署到 GitHub Pages，
约 1–2 分钟后线上更新。可在仓库 **Actions** 标签查看状态，绿勾即上线。

## 常见坑

1. **用了本地图**：PR 的 CI 会失败；本地 `serve.py` 或 `build.py --check` 能提前发现。
2. **误提交了 `site/papers.json`**：已 gitignore，别用 `git add -f` 强加。
3. **缺 `title`**：不会报错，但论文不会出现在网站上。
