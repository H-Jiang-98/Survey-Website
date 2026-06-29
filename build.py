#!/usr/bin/env python3
"""
build.py — System0 Survey 论文网站静态生成器

扫描当前目录下的 Obsidian markdown 笔记，提取元数据 / 缩略图 / 摘要 / 全文，
生成 site/papers.json 并把本地图片拷贝到 site/assets/。
前端 site/index.html 读取 papers.json 渲染卡片、筛选与全文弹窗。

用法:
    python3 build.py

只需重新运行本脚本即可在新增 / 修改笔记后更新网站数据。
"""

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS_SRC = ROOT / "assets"
SITE = ROOT / "site"
SITE_ASSETS = SITE / "assets"

# 这些是自动生成的目录页 (MOC)，不是论文笔记，跳过。
SKIP_NAMES = {"_inbox.md", "assets.md"}


# ----------------------------------------------------------------------------- 解析工具
def split_frontmatter(text):
    """返回 (frontmatter_dict, body_str)。无 frontmatter 时返回 ({}, text)。"""
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    import yaml

    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        fm = {}
    return fm, m.group(2)


def clean_wikilinks(s):
    """[[x|y]] -> y, [[x]] -> x。"""
    s = re.sub(r"\[\[([^\[\]|]+)\|([^\[\]]+)\]\]", r"\2", s)
    s = re.sub(r"\[\[([^\[\]]+)\]\]", r"\1", s)
    return s


def get_section(body, header):
    """提取某个 '## header' 标题下、直到下一个同级/更高级标题前的正文。"""
    pat = re.compile(
        r"^#{2,3}\s*" + re.escape(header) + r"\s*\n(.*?)(?=^#{1,3}\s|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    m = pat.search(body)
    return m.group(1).strip() if m else ""


def find_local_images(body):
    """返回正文中所有“非在线图”引用，用于合规校验。

    违规情形：
      - Obsidian 嵌入   ![[file.png]]            （永远是本地图）
      - 标准图片但非 http  ![alt](assets/x.png)   （指向仓库内相对路径）
    返回违规引用字符串列表；全为在线图时返回 []。
    """
    bad = []
    for m in re.finditer(r"!\[\[([^\[\]|]+?)(?:\|[^\[\]]*)?\]\]", body):
        bad.append(f"![[{m.group(1).strip()}]]")
    for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", body):
        src = m.group(1).strip()
        if not re.match(r"https?://", src):
            bad.append(f"![]({src})")
    return bad


def first_image(body):
    """返回正文中第一张图片 -> {'src':..., 'local':bool}。找不到返回 None。"""
    # 在文档中按出现顺序找 Obsidian 嵌入 ![[file|size]] 或标准 ![alt](url)
    embed = re.compile(r"!\[\[([^\[\]|]+?)(?:\|[^\[\]]*)?\]\]")
    md_img = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    best = None
    for m in embed.finditer(body):
        best = (m.start(), {"src": "assets/" + m.group(1).strip(), "local": True})
        break
    for m in md_img.finditer(body):
        cand = (m.start(), {"src": m.group(1).strip(), "local": False})
        if best is None or cand[0] < best[0]:
            best = cand
        break
    if best is None:
        return None
    # 取两类里位置更靠前的那张
    cands = []
    me = embed.search(body)
    mi = md_img.search(body)
    if me:
        cands.append((me.start(), {"src": "assets/" + me.group(1).strip(), "local": True}))
    if mi:
        cands.append((mi.start(), {"src": mi.group(1).strip(), "local": False}))
    cands.sort(key=lambda x: x[0])
    return cands[0][1]


def derive_arxiv_abs(text):
    """从任意 arxiv.org/{html,abs,pdf}/<id> 链接推出规范 abs 链接。"""
    m = re.search(r"arxiv\.org/(?:html|abs|pdf)/(\d{4}\.\d{4,5})", text)
    return f"https://arxiv.org/abs/{m.group(1)}" if m else ""


def extract_links(body, fm, img):
    """从『速查卡片』(变体A) 与『元信息』表格 (变体B) 和 frontmatter 提取论文链接。"""
    links = {}
    # 变体 A: 速查卡片  **label**: url
    label_map = {
        "代码": "code", "项目页": "project", "项目": "project",
        "arXiv": "arxiv", "论文": "paper", "项目主页": "project",
    }
    for zh, key in label_map.items():
        m = re.search(r"\*\*" + re.escape(zh) + r"\*\*[:：]\s*(\S+)", body)
        if m and m.group(1).strip().startswith("http"):
            links.setdefault(key, m.group(1).strip())

    # 变体 B: 元信息表格  | 项目主页 | url |  和  | 链接 | [arXiv](url) / [PDF](url) |
    for m in re.finditer(r"\|\s*项目主页\s*\|\s*([^\|]+?)\s*\|", body):
        cell = m.group(1).strip()
        um = re.search(r"https?://\S+", cell)
        if um:
            links.setdefault("project", um.group(0))
    # 任意 markdown 链接形式 [arXiv](url) / [代码](url) / [GitHub](url)
    mdlink = {"arxiv": r"arxiv", "code": r"代码|github|code"}
    for key, pat in mdlink.items():
        m = re.search(r"\[(?:" + pat + r")\]\((https?://[^)]+)\)", body, re.IGNORECASE)
        if m:
            links.setdefault(key, m.group(1))

    if fm.get("arxiv_html"):
        links.setdefault("arxiv_html", str(fm["arxiv_html"]))

    # 兜底: 从图片或已有链接里推规范 arXiv abs 链接
    if "arxiv" not in links:
        src = img["src"] if img else ""
        abs_url = derive_arxiv_abs(src) or derive_arxiv_abs(body) or derive_arxiv_abs(json.dumps(links))
        if abs_url:
            links["arxiv"] = abs_url
    return links


def primary_url(links):
    """卡片标题点击跳转的首选网址。"""
    for k in ("project", "arxiv", "paper", "arxiv_html", "code"):
        if links.get(k):
            return links[k]
    return ""


def preprocess_body(body):
    """把 Obsidian 专有语法转成标准 markdown，供前端 marked 渲染。"""
    # 嵌入图片 ![[file|size]] -> ![file](assets/file)（丢弃像素宽度，由 CSS 控制）
    body = re.sub(
        r"!\[\[([^\[\]|]+?)(?:\|[^\[\]]*)?\]\]",
        lambda m: f"![{m.group(1).strip()}](assets/{m.group(1).strip()})",
        body,
    )
    # 概念链接 [[x|y]] -> y, [[x]] -> x
    body = clean_wikilinks(body)
    # 标注块 > [!summary] Title -> > **📌 Title**
    body = re.sub(
        r"^>\s*\[!(\w+)\]\s*(.*)$",
        lambda m: f"> **📌 {m.group(2).strip() or m.group(1)}**",
        body,
        flags=re.MULTILINE,
    )
    return body.strip()


# ----------------------------------------------------------------------------- 主流程
def parse_paper(path):
    text = path.read_text(encoding="utf-8")
    fm, body = split_frontmatter(text)
    if not fm.get("title"):
        return None  # 不是论文笔记

    abstract = get_section(body, "一句话结论") or get_section(body, "一句话总结")
    abstract = clean_wikilinks(abstract)
    abstract = re.sub(r"(?m)^\s*>\s?", "", abstract)  # 去 blockquote 引用符
    abstract = re.sub(r"\n+-{3,}\s*$", "", abstract).strip()  # 去尾部 --- 分隔线

    img = first_image(body)
    links = extract_links(body, fm, img)

    return {
        "_violations": find_local_images(body),  # 图片合规校验（main 据此过滤），不写入 json
        "id": path.stem,
        "title": str(fm.get("title", path.stem)),
        "method": str(fm.get("method_name", path.stem)),
        "authors": fm.get("authors", []) or [],
        "year": fm.get("year", ""),
        "venue": str(fm.get("venue", "")),
        "tags": fm.get("tags", []) or [],
        "image": img,
        "abstract": abstract,
        "links": links,
        "url": primary_url(links),
        "body": preprocess_body(body),
    }


def validate_frontend_js():
    """语法校验 index.html 内联脚本（需要 esprima 时才检查，缺失则跳过）。

    曾因内联 JS 语法错误导致整页空白却无人发现——加这道软校验兜底。
    """
    index = SITE / "index.html"
    if not index.exists():
        return
    try:
        import esprima  # 可选依赖
    except ImportError:
        return
    html = index.read_text(encoding="utf-8")
    blocks = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
    for i, block in enumerate(blocks):
        try:
            esprima.parseScript(block, tolerant=False)
        except Exception as e:  # noqa: BLE001
            print(f"⚠ site/index.html 内联脚本 #{i+1} 语法错误：{e}")


def main():
    SITE.mkdir(exist_ok=True)
    if SITE_ASSETS.exists():
        shutil.rmtree(SITE_ASSETS)  # 清掉上次构建的残留图片，避免陈旧文件
    SITE_ASSETS.mkdir(exist_ok=True)

    papers = []
    skipped = []  # 图片不合规、被剔除的论文
    for path in sorted(ROOT.glob("*.md")):
        if path.name in SKIP_NAMES:
            continue
        paper = parse_paper(path)
        if not paper:
            continue
        violations = paper.pop("_violations", [])
        if violations:
            skipped.append((path.name, paper["method"], violations))
            continue  # 策略：图片必须全部为在线图，否则整篇剔除
        papers.append(paper)

    # 仅拷贝被纳入论文实际引用的本地图（当前策略下通常为空）
    copied = 0
    if ASSETS_SRC.is_dir():
        referenced = set()
        for p in papers:
            if p["image"] and p["image"]["local"]:
                referenced.add(Path(p["image"]["src"]).name)
        for png in ASSETS_SRC.glob("*"):
            if png.name in referenced:
                shutil.copy2(png, SITE_ASSETS / png.name)
                copied += 1

    # 警告：列出被剔除的不合规论文
    if skipped:
        print(f"\n⚠ 已剔除 {len(skipped)} 篇含本地图片的论文（要求图片必须为在线图）：")
        for name, method, viol in skipped:
            sample = ", ".join(viol[:3]) + (f" …等 {len(viol)} 处" if len(viol) > 3 else "")
            print(f"   ✗ {name}  ({method})  本地图: {sample}")
        print("   请提供方把这些图改为在线 URL 形式 ![](https://…) 后再纳入网站。\n")

    # 排序: 年份倒序, 再按 method 名
    papers.sort(key=lambda p: (-(int(p["year"]) if str(p["year"]).isdigit() else 0), p["method"].lower()))

    data = {
        "generated": "python3 build.py",
        "count": len(papers),
        "papers": papers,
    }
    (SITE / "papers.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 汇总可筛选维度供参考
    years = sorted({str(p["year"]) for p in papers if p["year"]})
    venues = sorted({p["venue"] for p in papers if p["venue"]})
    validate_frontend_js()

    print(f"✓ {len(papers)} 篇论文 -> site/papers.json（剔除 {len(skipped)} 篇）")
    print(f"✓ {copied} 张本地图片 -> site/assets/")
    print(f"  年份: {', '.join(years)}")
    print(f"  发表地: {', '.join(venues)}")


def lint_paths(paths):
    """校验给定 md 文件的图片是否全为在线图。返回违规列表 [(name, [refs])]。"""
    bad = []
    for p in paths:
        path = Path(p)
        if path.suffix.lower() != ".md" or path.name in SKIP_NAMES or not path.is_file():
            continue
        fm, body = split_frontmatter(path.read_text(encoding="utf-8"))
        if not fm.get("title"):
            continue  # 非论文笔记，跳过
        viol = find_local_images(body)
        if viol:
            bad.append((path.name, viol))
    return bad


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if args and args[0] == "--check":
        # 校验模式：只检查图片合规，不构建。供 CI 在 PR 上做门禁。
        targets = args[1:] or [str(p) for p in sorted(ROOT.glob("*.md"))]
        bad = lint_paths(targets)
        if bad:
            print("✗ 图片合规校验未通过：以下论文含本地图片，必须改为在线 URL ![](https://…)：")
            for name, viol in bad:
                sample = ", ".join(viol[:5]) + (f" …等 {len(viol)} 处" if len(viol) > 5 else "")
                print(f"   {name}: {sample}")
            sys.exit(1)
        print(f"✓ 图片合规校验通过（已检查 {len(targets)} 个文件）。")
        sys.exit(0)
    main()
