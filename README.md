# 论文综述网站

基于 Obsidian markdown 笔记自动生成的论文展示网站。

> 想添加一篇论文？请看 **[CONTRIBUTING.md](CONTRIBUTING.md)**（含完整提交流程与图片策略）。

## 使用

```bash
# 方式一：重建数据并自动启动本地服务器（推荐）
python3 serve.py

# 方式二：只重建数据，自己起服务器
python3 build.py
cd site && python3 -m http.server 8731
# 然后浏览器打开 http://localhost:8731
```

> 必须通过本地服务器访问（页面用 fetch 读取 `papers.json`，直接双击 `file://` 打开会被浏览器 CORS 拦截）。
> 公式与 markdown 渲染依赖 CDN（marked.js + KaTeX），首次加载需联网。

## 工作流

1. 在 `markdown_files/` 里新增 / 修改 `*.md` 笔记（图片必须为在线 URL，见 [CONTRIBUTING.md](CONTRIBUTING.md)）。
2. 跑一次 `python3 build.py`（或 `serve.py`）即可更新网站。

## 流水线做了什么（`build.py`）

- 扫描 `markdown_files/` 下的论文笔记（按名跳过可能出现的 Obsidian 目录页 `_inbox.md`/`assets.md`）。
- 从 frontmatter 提取 `title / method_name / authors / year / venue / tags`。
- 摘要：抓 `一句话结论` 或 `一句话总结` 段落。
- 缩略图：正文第一张关键图（在线 URL）。
- 论文网址：依次从『速查卡片』链接、『元信息』表格、或图片里的 arXiv id 推断。
- 全文：把 Obsidian 专有语法（`[[概念|别名]]` 链接、`> [!summary]` 标注块）转成标准 markdown。
- 图片合规校验：含本地图片的笔记会被剔除并警告（要求全部为在线图）。
- 输出 `site/papers.json`。

## 前端（`site/index.html`）

- 简洁卡片网格：关键图 + 标题 + 摘要 + 年份/发表地/关键词标签。
- **点标题** → 新标签打开论文网址；**点卡片其他区域** → 弹窗渲染完整笔记（含公式、图、表）。
- 顶部筛选栏：按 **年份 / 发表地 / 关键词** 聚合多选 + 全文搜索；点卡片上的关键词标签也能直接筛选。

## 部署到 GitHub Pages（自动）

仓库已带好 `.github/workflows/deploy.yml`：**每次 push（包括在 GitHub 网页上直接改 md）都会自动跑 `build.py` 并部署到 Pages**，解析逻辑只维护这一份 Python。

首次设置：

```bash
git init && git add -A && git commit -m "init"
git branch -M main
git remote add origin git@github.com:<你的用户名>/<仓库名>.git
git push -u origin main
```

然后在 GitHub 仓库页面：**Settings → Pages → Build and deployment → Source 选 “GitHub Actions”**。
之后访问 `https://<用户名>.github.io/<仓库名>/`。以后只要改 md 文件并 push，网站自动更新。

> 说明：`site/papers.json` 是构建产物，已在 `.gitignore` 里忽略，由 CI 每次重建。论文图片一律用在线 URL，仓库不存图片。页面所有路径都是相对路径，放在 `/<仓库名>/` 子路径下也能正常工作。

## 待办（已规划但未实现）

- arXiv 论文发表刊物的二次联网确认：可加一个 `venues.json` 覆盖表，`build.py` 合并后在卡片显示校验后的刊物。当前直接显示 frontmatter 里的 `venue`。
