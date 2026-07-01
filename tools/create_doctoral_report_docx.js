const fs = require("fs");
const path = require("path");
const {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  Header,
  HeadingLevel,
  ImageRun,
  LevelFormat,
  Packer,
  PageBreak,
  PageNumber,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableOfContents,
  TableRow,
  TextRun,
  VerticalAlign,
  WidthType,
} = require("docx");

const root = process.cwd();
const outDir = path.join(root, "reports");
const outPath = path.join(outDir, "WaveST-Gate_博士汇报逐字稿_详细版.docx");

fs.mkdirSync(outDir, { recursive: true });

const font = "Microsoft YaHei";
const page = {
  width: 11906,
  height: 16838,
  margin: { top: 960, right: 900, bottom: 900, left: 900 },
};
const tableWidth = page.width - page.margin.left - page.margin.right;

function run(text, opts = {}) {
  return new TextRun({
    text,
    font,
    size: opts.size || 22,
    bold: !!opts.bold,
    italics: !!opts.italics,
    color: opts.color || "111111",
    break: opts.break,
  });
}

function p(text, opts = {}) {
  return new Paragraph({
    alignment: opts.alignment || AlignmentType.LEFT,
    spacing: { before: opts.before || 0, after: opts.after || 120, line: opts.line || 330 },
    indent: opts.indent ? { left: opts.indent } : undefined,
    border: opts.border,
    children: [run(text, opts.run || {})],
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 320, after: 180 },
    children: [run(text, { bold: true, size: 34, color: "12324A" })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 260, after: 140 },
    children: [run(text, { bold: true, size: 28, color: "1E5A68" })],
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 100 },
    children: [run(text, { bold: true, size: 24, color: "7A3150" })],
  });
}

function note(text) {
  return new Paragraph({
    spacing: { before: 60, after: 120, line: 300 },
    shading: { fill: "EEF6F7", type: ShadingType.CLEAR },
    border: {
      left: { style: BorderStyle.SINGLE, size: 12, color: "2E9AA5" },
      top: { style: BorderStyle.SINGLE, size: 1, color: "D3E6EA" },
      bottom: { style: BorderStyle.SINGLE, size: 1, color: "D3E6EA" },
    },
    children: [run(text, { size: 21, color: "12324A" })],
  });
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

function numbered(text, reference = "numbers") {
  return new Paragraph({
    numbering: { reference, level: 0 },
    spacing: { after: 90, line: 320 },
    children: [run(text, { size: 21 })],
  });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80, line: 310 },
    children: [run(text, { size: 21 })],
  });
}

function cell(text, width, opts = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    verticalAlign: VerticalAlign.TOP,
    shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 1, color: "C7D3D9" },
      bottom: { style: BorderStyle.SINGLE, size: 1, color: "C7D3D9" },
      left: { style: BorderStyle.SINGLE, size: 1, color: "C7D3D9" },
      right: { style: BorderStyle.SINGLE, size: 1, color: "C7D3D9" },
    },
    margins: { top: 100, bottom: 100, left: 120, right: 120 },
    children: [
      new Paragraph({
        spacing: { after: 0, line: 300 },
        children: [run(text, { size: opts.size || 20, bold: opts.bold, color: opts.color || "111111" })],
      }),
    ],
  });
}

function table(rows, widths) {
  return new Table({
    width: { size: tableWidth, type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((row, i) =>
      new TableRow({
        children: row.map((txt, j) =>
          cell(txt, widths[j], {
            fill: i === 0 ? "DDEFF2" : undefined,
            bold: i === 0,
            color: i === 0 ? "12324A" : "111111",
          })
        ),
      })
    ),
  });
}

function imageParagraph(rel, widthPx, title) {
  const abs = path.join(root, rel);
  if (!fs.existsSync(abs)) {
    return note(`图片文件未找到：${rel}`);
  }
  const dims = {
    "figure_1_editorial_graphical_abstract.png": [3600, 2025],
    "figure_1_workflow_schematic.png": [3600, 3650],
    "figure_2_spatial_cell_composition.png": [3600, 2400],
    "figure_3_baseline_performance.png": [3600, 6000],
    "figure_4_reliability_calibration.png": [3600, 2800],
    "figure_5_boundary_niche_pathology.png": [3600, 2400],
    "supplementary_figure_s1_robustness.png": [3200, 2100],
  };
  const base = path.basename(rel);
  const [w, h] = dims[base] || [1600, 1000];
  const heightPx = Math.round((widthPx * h) / w);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 160 },
    children: [
      new ImageRun({
        type: "png",
        data: fs.readFileSync(abs),
        transformation: { width: widthPx, height: heightPx },
        altText: { title, description: title, name: title },
      }),
    ],
  });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 220, line: 280 },
    children: [run(text, { size: 18, italics: true, color: "4D5A61" })],
  });
}

const children = [];

children.push(
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 260, after: 120 },
    children: [run("WaveST-Gate 博士汇报逐字稿", { bold: true, size: 40, color: "12324A" })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 280 },
    children: [run("Xenium 监督的乳腺癌空间反卷积基准、可靠性门控模型与病理验证证据链", { size: 25, color: "1E5A68" })],
  }),
  imageParagraph("results/nature_manuscript_figures/figure_1_editorial_graphical_abstract.png", 610, "Graphical abstract"),
  note("使用方式：这份稿子按 25 到 35 分钟博士汇报设计，语气尽量保持自然、克制、像自己在解释一项研究。正文可以直接照读；每张图后面的“我会这样讲”可以作为换图时的口播稿。"),
  table(
    [
      ["汇报核心句", "WaveST-Gate 的核心贡献不是只提高一个反卷积指标，而是建立了 Xenium 监督的 Visium 级真实标签基准，并把准确性、可靠性、组织边界、生物 niche、外部病理一致性和可复现发布连成完整证据链。"],
      ["一句话创新", "用高分辨率 Xenium 单细胞注释构建可评估的空间反卷积 ground truth，再用 morphology-aware、reference-grounded、reliability-calibrated 的多模态模型进行细胞组成推断。"],
      ["主要结果", "主模型 JSD 为 0.01293，spotwise cosine 为 0.9933，在 13 个方法中排名第一；边界跳变比为 2.36x，识别 5 类组织 niche，外部病理一致性为 0.82。"],
    ],
    [2200, tableWidth - 2200]
  ),
  pageBreak()
);

children.push(
  h1("一、正式开场稿"),
  p("各位老师好，我今天汇报的工作是 WaveST-Gate，一个面向乳腺癌空间转录组反卷积的多模态方法。这个工作的出发点很直接：空间转录组给了我们组织位置和表达信息，H&E 给了我们病理形态，而单细胞或空间单细胞数据给了我们更高分辨率的细胞类型注释。问题是，过去很多方法要么缺少可量化的真实标签，要么只在表达层面做反卷积，很难回答两个关键问题：预测到底准不准，以及这些预测在组织和病理层面是否可信。"),
  p("所以我这项工作的核心不是单纯再提出一个模型结构，而是先把评估问题本身做扎实。我们把 Xenium 细胞级注释映射到 Visium/CytAssist spot 邻域里，构建了一个可复现、可审计的 Xenium-to-Visium benchmark。这个 benchmark 里有 4,992 个 spots，其中 485 个 spots 有 Xenium-derived ground truth，覆盖 115,275 个已注释 Xenium 细胞和 19 个细胞类型，聚合半径固定为 55。"),
  p("在这个基准上，我们提出 WaveST-Gate。模型把三类信息放在同一个框架里：第一是 H&E 图像，用 wavelet-guided morphology encoder 表征低频组织背景和高频边界纹理；第二是 spot-level gene expression；第三是 scRNA-derived cell-type prototypes，也就是来自参考单细胞数据的细胞类型先验。随后模型通过 reference-aware agents 和 cross-modal reliability gate，把这些信息融合成 spot-level 的细胞比例、表达重构、不确定性、注意力和 niche 输出。"),
  p("我想强调的是，这里的 gate 不是普通意义上为了加权融合而加的 attention。我们希望它承担 reliability 的含义，也就是告诉我们在不同组织区域里，模型更依赖 H&E、表达还是参考先验，哪些点更可信，哪些点需要谨慎解释。因此后面的结果不仅看 JSD、cosine 这些准确性指标，还会系统验证 calibration、risk coverage、boundary preservation、niche interpretation 和 external pathology correspondence。"),
  p("整个汇报我会按四个层次展开。第一，为什么需要这样一个 Xenium 监督的 benchmark；第二，WaveST-Gate 的模型设计和训练目标；第三，主结果、baseline 比较和消融实验；第四，可靠性、边界、niche、外部泛化和稳健性证据。最后我会总结这项工作的创新点和可能的局限。")
);

children.push(
  h1("二、研究背景和问题定义"),
  h2("1. 为什么要做 spatial deconvolution"),
  p("空间转录组的一个核心任务，是从每个 spot 的混合表达里恢复细胞类型组成。对乳腺癌这样的组织来说，这件事尤其重要，因为肿瘤区域、基质区域、免疫浸润、导管结构和坏死区域往往不是随机分布的，而是和病理状态、局部微环境以及治疗相关生物学密切联系在一起。"),
  p("传统 deconvolution 方法通常依赖表达矩阵和单细胞参考，比如 RCTD、cell2location、Tangram、CARD、BayesPrism 等。这些方法各有优势，但在实际病理组织中会遇到几个问题：spot 内细胞混合复杂，组织边界会带来突然变化；H&E 中有明显形态信息，但很多方法没有充分利用；更关键的是，我们很难得到 spot-level 的真实细胞组成，因此很多结果只能看起来合理，却很难严谨比较。"),
  h2("2. 这个领域的主要痛点"),
  numbered("第一个痛点是真实标签不足。没有真实标签时，模型只能靠间接指标或可视化判断，很难说谁真的更准确。"),
  numbered("第二个痛点是比较不够公平。不同方法可能使用不同基因、不同参考、不同 spot 子集，导致 baseline comparison 变成实现细节的比较，而不是方法能力的比较。"),
  numbered("第三个痛点是形态学没有被系统纳入。H&E 里有肿瘤边界、导管结构、免疫浸润和组织坏死等线索，但如果只是把图像特征简单拼接到表达特征上，很难证明图像真的贡献了可解释信息。"),
  numbered("第四个痛点是可信度不足。模型给出一个比例并不够，我们还需要知道这个比例在什么区域可信，在什么区域可能出错，是否有不确定性和校准证据支持。"),
  numbered("第五个痛点是生物解释和外部验证不足。即使模型指标好，也要进一步回答这些预测能否对应组织边界、niche 状态和外部病理分类。"),
  h2("3. 我们的总体思路"),
  p("针对这些问题，我们没有把工作只定义成“做一个更复杂的网络”。我们的思路是从 ground truth、模型、评估和生物验证四个环节一起做。ground truth 上，我们用 Xenium 单细胞级注释构建 Visium 级 spot label；模型上，我们设计 morphology-aware 和 reference-grounded 的融合框架；评估上，我们用强 baseline、bootstrap、paired permutation、消融和敏感性分析形成闭环；解释上，我们进一步看可靠性、边界、niche 和外部病理对应。")
);

children.push(
  h1("三、我们具体做了什么工作"),
  h2("1. 构建 Xenium-to-Visium benchmark"),
  p("第一部分工作是构建 benchmark。我们将 Xenium 中带有细胞类型注释的单细胞位置，与 Visium/CytAssist spot 坐标对齐。然后以每个 spot 为中心，在固定半径 55 的邻域内统计 Xenium 细胞，得到每个 spot 的细胞类型计数和比例。这个过程产生了 counts、proportions、spot-level QC、entropy、dominant cell type、spatial split 和 manifest 等独立文件。"),
  p("这个设计的意义在于，它把原本很难评估的空间反卷积问题，转化成了一个可以被监督评估的 benchmark。我们不是只用视觉直觉说模型预测合理，而是可以直接比较预测比例和 Xenium-derived ground truth 的距离。主 benchmark 有 4,992 个 spots，其中 485 个有 ground truth，涉及 115,275 个 Xenium 细胞和 19 个细胞类型。"),
  h2("2. 设计 WaveST-Gate 模型"),
  p("第二部分是模型设计。WaveST-Gate 的输入包括 H&E patch、spot expression 和 scRNA reference prototype。H&E 分支不是普通 CNN，而是用 wavelet-guided morphology encoder 区分低频组织背景和高频方向性边界纹理。这样做的直觉是，乳腺癌组织中的肿瘤-基质边界、导管轮廓和免疫浸润边缘常常体现在形态纹理和边界结构上。"),
  p("表达分支用 MLP 编码 spot-level gene expression。参考分支把 scRNA-derived cell-type prototypes 变成 cell-type agents，让模型在推断每个 spot 的组成时，不是只靠局部表达，而是有细胞类型先验参与。随后 cross-modal reliability gate 学习 H&E、expression 和 reference 三类信息在不同 spot 上的可靠性权重。"),
  p("输出层包括 predicted proportions、reconstructed expression、spot uncertainty、agent attention 和 niche logits。也就是说，模型不仅输出“这个 spot 有多少肿瘤细胞、多少基质细胞”，还输出“这个预测是否可信，以及它可能属于哪类组织状态”。"),
  h2("3. 训练目标和评估指标"),
  p("训练目标包括表达重构、Xenium-supervised proportion loss、entropy/sparsity regularization、spatial smoothness、boundary-aware constraints、contrastive alignment 和 uncertainty calibration。主模型训练到 step 499，得到 JSD 0.01293、spotwise cosine 0.9933、mean cell-type Pearson 0.9283、expression log1p RMSE 0.6379。"),
  p("评估指标上，我们没有只看单一 JSD。主指标包括 JSD、spotwise cosine、mean cell-type Pearson 和 RMSE；baseline 评估里还记录 runtime、peak memory、bootstrap mean/std 和 paired permutation；可靠性部分看 uncertainty-error correlation、calibration bins 和 risk coverage；生物学部分看 boundary jump、niche marker enrichment 和 external pathology agreement。"),
  h2("4. 做强 baseline 和公平比较"),
  p("第三部分是 baseline comparison。我们把 WaveST-Gate 和 12 个 baseline 放在同一比较框架下，包括 BayesPrism、RCTD multi、SpatialDWLS/Seurat、CARD、reference cosine、reference NNLS、SpatialDWLS、Tangram、cell2location、uniform 和 SPOTlight 等。比较时尽量保证 shared supervised spots、shared genes/reference where applicable 和 common metrics。"),
  p("最终 WaveST-Gate 在 13 个方法中按 JSD 排名第一，JSD 为 0.01293；最强的非 WaveST-Gate baseline 是 BayesPrism，JSD 约为 0.2377。这个差距不是只靠单次结果展示，而是通过 bootstrap、paired statistics 和审稿控制一起支撑。"),
  h2("5. 做消融、可靠性、边界、niche、外部和稳健性实验"),
  p("第四部分是证据链。我们做了 12 组消融，包括去掉 wavelet、去掉 image branch、去掉 cell-type agents、用 mean fusion 替代 gate、去掉 uncertainty、去掉 boundary loss、去掉 local refinement，以及 expression-only、image-only、reference-only 等。这样可以回答模型每个模块是不是必要。"),
  p("随后我们做可靠性分析，证明 uncertainty 与 spot-level error 有相关性，calibration-bin Pearson 达到 0.9619。边界分析中，boundary-to-interior jump ratio 为 2.36，说明模型能够保留组织边界而不是简单过平滑。niche 分析中，我们识别了 5 类 biological niches，并通过 marker enrichment、Xenium neighborhood 和 external pathology correspondence 验证。最后，在外部 no-retuning、Rep1 minimal retuning、输入扰动、split sensitivity 和 aggregation sensitivity 上做稳健性检查。")
);

children.push(
  h1("四、创新点总结"),
  h2("创新点一：把 Xenium 单细胞信息转化为 Visium 级可评估 benchmark"),
  p("我认为这项工作的第一个创新点是 benchmark 层面的。很多空间反卷积工作最大的困难不是模型不够复杂，而是缺少可以严谨评估的真实标签。我们用 Xenium 细胞级注释构建 Visium spot-level ground truth，使得空间反卷积可以在真实乳腺癌组织上进行监督评估。这个 benchmark 不是一次性临时处理，而是包含 manifest、QC、split、sensitivity grid 和 protocol 的可复现对象。"),
  h2("创新点二：用 wavelet morphology 显式建模 H&E 中的组织边界和纹理"),
  p("第二个创新点是形态学建模。我们没有简单地把 H&E patch 丢进普通 CNN，而是用 wavelet-guided morphology encoder 分离低频组织背景和高频方向性边界。这样模型更容易捕捉肿瘤-基质边界、导管结构、免疫边缘等病理形态线索。这个设计后面通过 image-gate control、texture-stratified improvement 和 boundary preservation 进行验证。"),
  h2("创新点三：把 scRNA 参考变成 cell-type prototype agents"),
  p("第三个创新点是 reference-grounded agents。传统方法常常把单细胞参考作为静态表达模板，而我们把 cell-type prototypes 变成 agents，让它们参与 spot-level 表征和比例预测。这样模型可以把局部 ST 表达、H&E 形态和参考细胞类型先验放在同一个推断空间里，而不是后处理式地拼接。"),
  h2("创新点四：cross-modal reliability gate 不是普通 attention，而是可信度建模"),
  p("第四个创新点是 reliability gate。我们希望 gate 不只是告诉模型哪个模态权重大，而是形成一种可解释的 trust state。Figure 4 里我们用 uncertainty-error correlation、calibration bins、risk coverage 和 H&E local support 来证明 gate 的可靠性语义。对实际应用来说，这一点很重要，因为病理空间图谱不仅要预测，还要知道哪里可信。"),
  h2("创新点五：从准确性推进到边界、niche 和外部病理一致性"),
  p("第五个创新点是验证体系。我们没有停在 JSD 排名第一，而是继续问预测是否保留组织边界，是否形成有生物意义的 tumor-immune-stromal niches，是否能和外部病理分类对应。Figure 5 中 boundary jump、marker support、niche identity 和 external pathology agreement 一起说明模型学到的不是纯数学拟合，而是可以解释的组织状态。"),
  h2("创新点六：提交级别的完整证据链"),
  p("最后一个创新点是工程和投稿层面的完整性。我们准备了 release bundle、Zenodo DOI、environment report、availability statements、figure/table manifests、readiness audit 和 reviewer preflight。这个工作不只是一个实验结果，而是围绕 Nature 级别投稿要求整理成了一套可审计证据包。")
);

children.push(pageBreak(), h1("五、逐图汇报稿"));

children.push(
  h2("Graphical Abstract：一眼讲完整故事"),
  imageParagraph("results/nature_manuscript_figures/figure_1_editorial_graphical_abstract.png", 610, "Graphical abstract"),
  caption("Graphical Abstract：从输入、模型到输出和证据墙的一屏概览。"),
  p("这张图我会放在开头，用来建立听众对整个工作的第一印象。左侧是输入，包括 H&E 组织形态、Xenium cells 和空间表达；中间是模型，也就是 reliability-gated fusion；右侧是输出，包括 deconvolved TME atlas 和 external pathology validation。下面这一排 evidence wall 则把文章最重要的结果压缩成几个数字：13 个 baseline 中排名第一，主 JSD 0.01293，spot cosine 0.9933，外部病理 spots 15,601，DOI 已发布。"),
  note("我会这样讲：这张图概括了我们工作的主线。我们从 H&E、空间转录组和 Xenium 单细胞监督出发，设计了一个 reliability-gated 的多模态模型，最后输出空间细胞组成图谱，并用 benchmark、baseline、病理验证和可复现发布来支撑。")
);

children.push(
  h2("Figure 1：方法和 benchmark 的总览图"),
  imageParagraph("results/nature_manuscript_figures/figure_1_workflow_schematic.png", 610, "Figure 1"),
  caption("Figure 1：Xenium-to-Visium benchmark 构建与 WaveST-Gate 工作流。"),
  p("Figure 1 是整篇文章的方法总图。它回答两个问题：第一，我们的 ground truth 从哪里来；第二，WaveST-Gate 如何把不同模态的信息整合起来。图的上半部分展示了 H&E、ST expression、scRNA reference 和 typed Xenium cells 如何进入模型。Xenium cells 被聚合到 Visium spot 邻域，形成 spot-level ground truth。模型部分包括 wavelet morphology、expression latent、cell-type agents、trust gate 和 local refinement。输出部分包括细胞比例图谱、image/reference gate、uncertainty、boundary 和 composition。"),
  p("图中间和下方的 study assets、batch objects、release objects 和 annotation objects 说明这不是一个只在内存里跑出来的实验，而是一个完整数据对象和软件包。这里我会强调 4 个数字：4,992 个 spots，485 个 ground-truth spots，115,275 个 Xenium cells，19 个 cell types。"),
  note("我会这样讲：Figure 1 的重点是可信基准。我们先把高分辨率 Xenium 细胞注释转成 Visium spot 级 ground truth，然后再训练模型。这样后面的性能比较不是主观判断，而是有真实标签支撑。"),
  p("这张图在文章里的作用是打地基。听众如果接受了 Figure 1，就会理解后面的所有结果不是孤立的可视化，而是建立在可复现 benchmark 上的系统评估。")
);

children.push(
  h2("Figure 2：模型生成的空间细胞组成 atlas"),
  imageParagraph("results/nature_manuscript_figures/figure_2_spatial_cell_composition.png", 610, "Figure 2"),
  caption("Figure 2：H&E 支撑下的肿瘤微环境空间图谱、概率场和细胞状态结构。"),
  p("Figure 2 是主结果图，重点不是模型结构，而是模型到底在真实组织上恢复了什么空间生物学。上半部分的 whole-tissue TME atlas 把预测的 dominant compartment 叠加在 H&E 上。这里可以看到 tumor、stroma、immune 和 ductal 等区域不是随机分布的，而是和组织形态有明显对应。右侧的 atlas grammar 把整体空间组织简化成几类 compartment 和 niche anchors，便于从整体上理解组织状态。"),
  p("下半部分展示更细的 probability fields。比如 tumor field、immune field、stroma field 和 ductal field，不是简单给每个 spot 一个离散标签，而是展示不同细胞群在组织中的连续概率分布。中间的 cell-state heat wall 则把细胞类型、biological group 和 niche 联系起来，右侧 abundance grammar 用更紧凑的方式概括细胞丰度。"),
  p("这张图里我会强调三个指标：主模型 JSD 为 0.01293，spotwise cosine 为 0.9933，mean cell-type Pearson 为 0.9283。这些指标说明模型预测与 Xenium-derived ground truth 高度一致。同时，这张图从视觉上说明模型能恢复乳腺癌组织中的 TME 空间结构。"),
  note("我会这样讲：Figure 2 证明 WaveST-Gate 不只是表格上指标好，它能在整张组织切片上恢复肿瘤、基质、免疫和导管区域的空间组织，并把这些结果整理成可解释的 atlas。"),
  p("这张图在文章中的作用是把方法转化为生物图谱。它告诉读者，模型输出不仅可以量化评估，也可以作为组织微环境分析的空间基础。")
);

children.push(
  h2("Figure 3：公平 baseline 比较和审稿控制"),
  imageParagraph("results/nature_manuscript_figures/figure_3_baseline_performance.png", 430, "Figure 3"),
  caption("Figure 3：固定 benchmark 下的综合性能墙。"),
  p("Figure 3 是性能证据最密集的一张图。汇报时不要逐格念，而要按逻辑讲：先固定比较规则，再展示排名，再展示空间一致性，再展示外部适应和消融，最后展示审稿控制。"),
  p("图的左上角 benchmark contract 说明所有方法都在同一 Xenium-derived spot ground truth 下比较。这个部分很重要，因为 spatial deconvolution 的 baseline 很容易因为输入、基因、参考或 spot 子集不同而不公平。这里我们尽量把可比较条件固定下来。"),
  p("左中部分是 primary rank cliff。WaveST-Gate 在 13 个方法中按 JSD 排名第一，主 JSD 为 0.01293。最强非 WaveST-Gate baseline 是 BayesPrism，JSD 约 0.2377，所以图里给出约 18.4 倍的 gap。这个结果说明 WaveST-Gate 在 matched Xenium-supervised spots 上有非常明显的误差优势。"),
  p("中间还有 spatial concordance 面板，展示 Xenium ground truth 与 WaveST-Gate 预测在同一 Visium lattice 上的空间一致性。这里不是只看一个数字，而是看 spatial pattern 是否对得上。"),
  p("图的下半部分是 transfer、mechanism 和 reviewer controls。Rep1 direct no-retuning 被诚实地作为 domain-shift case 报告，而 minimal retuning budget curve 说明只需 25 steps 就能超过 Rep1 上最强 baseline。消融部分则显示 gate、boundary、agents 和 wavelet 等模块的必要性。审稿控制里包括 split stability、label-null separation、bootstrap dominance 和 input fairness lock，主要是为了回答“是不是 split 偶然、label 泄漏、baseline 不公平”这类问题。"),
  note("我会这样讲：Figure 3 是文章的性能主证据。我们先固定公平 benchmark，然后和 13 个方法比较。WaveST-Gate 排名第一，JSD 为 0.01293，并且优势不是单点结果，而是通过 bootstrap、paired statistics、消融、外部适应和输入公平性检查共同支撑。"),
  p("这张图的作用是说服审稿人：我们的优势不是因为选了一个弱 baseline，也不是因为只挑一个好看的指标，而是在严格比较和多层控制下仍然成立。")
);

children.push(
  h2("Figure 4：可靠性、校准和 trust state"),
  imageParagraph("results/nature_manuscript_figures/figure_4_reliability_calibration.png", 610, "Figure 4"),
  caption("Figure 4：把多模态不确定性转化为可解释的可靠性状态。"),
  p("Figure 4 解决的问题是：模型预测出来以后，我们怎么知道哪里可信。空间图谱在生物学应用中不能只给一个比例，还需要告诉用户哪些区域可以放心解释，哪些区域应该作为高风险或需要复核的区域。"),
  p("上半部分是 reliability-state atlas。它把 low、intermediate 和 high/review trust states 叠加到真实 H&E 组织形态上，同时标出局部 H&E support 和 error hot spots。右侧 risk transition grammar 把连续的 JSD 风险转成 low、intermediate、high 三类 trust states，便于实际使用。"),
  p("下半部分把可靠性和 Xenium-supervised error 对齐。这里最关键的结果是 uncertainty-error Pearson 约 0.5263，calibration-bin Pearson 约 0.9619。前者说明模型的不确定性和实际错误有中等相关性，后者说明按不确定性分箱后，预测风险和观测误差趋势高度一致。换句话说，模型不确定性不是装饰性的输出，而是能反映真实风险。"),
  p("右侧 biological trust proof wall 把 niche、modality 和 trust flow 联系起来，说明不同 niche 和不同模态贡献之间存在结构化关系。这也回应了一个潜在质疑：gate 是不是普通 attention。我们的回答是，gate 被放在 uncertainty、calibration 和 tissue state 的共同证据里解释，因此它具有 reliability semantics。"),
  note("我会这样讲：Figure 4 的核心是可信度。WaveST-Gate 不只预测细胞比例，还能输出空间化的 trust state。模型的不确定性与真实错误相关，并且 calibration bins 呈现很强的一致趋势，所以这套输出更适合后续病理和生物解释。"),
  p("这张图在文章里的作用是把模型从“准确”推进到“可被信任”。这是博士汇报中很重要的层次，因为它体现出我们不是只追求指标，而是在考虑实际科研使用中如何判断结果可靠性。")
);

children.push(
  h2("Figure 5：组织边界、niche 和外部病理对应"),
  imageParagraph("results/nature_manuscript_figures/figure_5_boundary_niche_pathology.png", 610, "Figure 5"),
  caption("Figure 5：H&E 边界到 biological niche，再到外部病理验证的机制链。"),
  p("Figure 5 是整篇文章的生物解释图。它的逻辑是从 H&E boundary 开始，到 spatial biological niche atlas，再到 external pathology correspondence。也就是说，我们不仅证明模型预测准确，还证明这些预测和组织结构、细胞状态以及外部病理标签有一致关系。"),
  p("左侧 H&E boundary evidence 显示模型在组织边界处保留了更明显的 composition jump。核心数字是 boundary-to-interior jump ratio 约 2.36x，typed edges 为 1,074。这说明模型没有把组织空间结构过度平滑掉，而是在 tumor-stroma、ductal、immune edge 等区域保留了边界变化。"),
  p("中间 spatial biological niche atlas 把组织划分为 5 个 niche。这里的 niche 不是随意聚类得到一个颜色图，而是结合细胞组成、marker enrichment、reference-agent support 和 H&E/pathology 关系来解释。下方的 marker enrichment 和 niche identity 面板展示每个 niche 对应的细胞状态特征。"),
  p("右侧 external pathology correspondence 是外部验证。我们把外部 Wu/Swarbrick 等数据中的病理分类和预测 niche 做对应，外部 spots 数量为 15,601，mean pathology agreement 约 0.82。这个结果说明预测 niche 与外部病理类别之间有较高一致性。"),
  note("我会这样讲：Figure 5 的重点是从机制到验证。H&E 边界帮助模型保留组织边缘，模型进一步形成可解释的 biological niches，而这些 niche 又能和外部病理分类对应。这样我们的结果就不只是数值准确，而是有病理和生物学意义。"),
  p("这张图的作用是完成文章的生物学闭环。Figure 3 证明模型强，Figure 4 证明模型可信，Figure 5 证明模型结果有组织病理意义。")
);

children.push(
  h2("Supplementary Figure S1：稳健性、敏感性和外部适应"),
  imageParagraph("results/nature_manuscript_figures/supplementary_figure_s1_robustness.png", 610, "Supplementary Figure S1"),
  caption("Supplementary Figure S1：输入扰动、split、半径敏感性和 Rep1 适应曲线。"),
  p("S1 是补充图，但答辩和投稿时非常重要，因为它回答结果是不是脆弱。左上角展示 stress-test ranking，包括 clean、subgroup、H&E perturbation、prototype perturbation、reference missing cell type、patch size、split、gene dropout 和 gene panel 等情形。"),
  p("右上角 radius/cell-count confidence landscape 展示 benchmark construction 是否依赖某一个任意半径或最低细胞数阈值。这里我们检查了 radii 45、55、65、75，以及 minimum cell count 1、5、10、20、50。主设置 radius 55、min cell 1 被明确标注，JSD 为 0.01293，supervised spots 为 485。"),
  p("中间的 GT split accounting 用来解释 primary validation split 没有 supervised GT 的问题。我们保留原始 spatial holdout 作为预设 benchmark，同时用 GT-stratified split sensitivity 做补充，证明 supervised validation/test spots 存在时结果仍然稳定。"),
  p("右侧 Rep1 retuning curve 是外部 domain shift 的诚实报告。直接 no-retuning 的 Rep1 JSD 是 0.3656，不优于最强 baseline；但 minimal retuning 只需要 25 steps 就能把 JSD 降到 0.1259，并超过最佳 baseline。这说明跨样本 domain shift 是真实存在的，但模型可以用很小预算完成适应。"),
  note("我会这样讲：S1 说明我们的结果不是单一参数、单一 split 或单一输入条件下的偶然结果。我们检查了输入扰动、半径和 cell-count 阈值、GT split、外部样本适应等多个维度，结果整体支持模型的稳定性。")
);

children.push(pageBreak(), h1("六、整段可照读的 8 分钟压缩版"));
[
  "如果时间比较短，我会这样汇报：各位老师好，我今天汇报 WaveST-Gate。这个工作的核心目标，是在乳腺癌空间转录组中做更可信的细胞组成反卷积。我们面对的主要问题是，很多空间反卷积方法缺少真实 spot-level 标签，baseline 比较很难公平，同时 H&E 形态信息和模型不确定性往往没有被系统利用。",
  "为了解决这个问题，我们首先构建了 Xenium-to-Visium benchmark。具体来说，我们把 Xenium 中带细胞类型注释的单细胞位置聚合到 Visium/CytAssist spot 邻域，形成 spot-level ground truth。这个 benchmark 包含 4,992 个 spots、485 个有 Xenium ground truth 的 spots、115,275 个 Xenium 细胞和 19 个细胞类型，聚合半径固定为 55。这样我们就可以在真实乳腺癌组织上对反卷积结果进行定量评估。",
  "在模型上，我们提出 WaveST-Gate。它同时使用 H&E 图像、spot expression 和 scRNA reference。H&E 分支用 wavelet morphology encoder 捕捉低频组织背景和高频边界纹理；表达分支编码 spot transcriptome；参考分支把单细胞原型变成 cell-type agents。随后 cross-modal reliability gate 学习不同 spot 上图像、表达和参考信息的可靠性权重，输出细胞比例、表达重构、不确定性和 niche 状态。",
  "主结果上，WaveST-Gate 的 JSD 为 0.01293，spotwise cosine 为 0.9933，mean cell-type Pearson 为 0.9283。在 13 个方法的公平 baseline comparison 中，WaveST-Gate 排名第一；最强非本方法 baseline 是 BayesPrism，JSD 约 0.2377。我们还做了 bootstrap、paired statistics、runtime、memory 和 input fairness controls，避免比较不公平。",
  "除了准确性，我们重点验证了可靠性和生物解释。Figure 4 显示模型 uncertainty 与真实 error 有相关性，uncertainty-error Pearson 为 0.5263，calibration-bin Pearson 为 0.9619，说明模型能够给出有意义的 trust state。Figure 5 显示模型保留组织边界，boundary-to-interior jump ratio 为 2.36，并识别 5 个 biological niches，这些 niche 与 15,601 个外部 pathology spots 的病理分类有 0.82 的一致性。",
  "最后，我们做了 12 个消融、外部 no-retuning 和 minimal-retuning、gene dropout、gene panel、reference mismatch、H&E perturbation、patch size、split sensitivity 和 radius/cell-count sensitivity 等稳健性分析。整体结果说明，WaveST-Gate 的贡献不是单纯在一个指标上更好，而是建立了一个 Xenium 监督的 benchmark，并把准确性、可靠性、组织边界、生物 niche、外部病理验证和可复现发布连成了完整证据链。"
].forEach((txt) => children.push(p(txt)));

children.push(pageBreak(), h1("七、可能被问到的问题和回答"));
[
  [
    "问：为什么要自己构建 Xenium-to-Visium benchmark？",
    "答：因为空间反卷积最大的问题之一是缺少真实 spot-level 标签。Xenium 有单细胞级空间位置和注释，Visium 有 spot-level 表达和 H&E 形态。我们把 Xenium 细胞聚合到 Visium spot 邻域，相当于把高分辨率细胞信息转化成 Visium 级监督信号。这样才能公平地比较不同反卷积方法，而不是只看可视化是否好看。"
  ],
  [
    "问：半径 55 会不会是任意选择？",
    "答：我们没有只依赖一个半径。主分析固定 radius 55 作为预设 benchmark，同时做了 radius 45、55、65、75 和最低 cell-count 阈值 1、5、10、20、50 的敏感性分析。S1 中可以看到覆盖率和性能随半径变化的情况。选择 radius 55 是在空间尺度和 Xenium cell coverage 之间的折中，并且有 sensitivity evidence 支撑。"
  ],
  [
    "问：WaveST-Gate 和普通 CNN 加 attention 有什么区别？",
    "答：区别主要在三个层面。第一，H&E 分支不是普通 CNN，而是 wavelet-guided morphology encoder，显式区分组织背景和方向性边界纹理。第二，reference 不是简单拼接，而是 cell-type prototype agents 参与推断。第三，gate 不只是 attention 权重，而是和 uncertainty、calibration、risk coverage 共同验证的 reliability gate。"
  ],
  [
    "问：H&E 图像真的有贡献吗？",
    "答：主模型的平均 image gate 不一定很高，因为很多 expression-rich spots 本身已经很有信息。但我们专门做了 image-gate-enhanced control 和 matched no-image control。结果显示 image-enhanced run 的 JSD 低于 no-image control，而且在 high-texture spots 中 paired improvement 更明显。这说明 H&E 的贡献主要体现在边界和高形态信息区域。"
  ],
  [
    "问：baseline comparison 是否公平？",
    "答：我们尽量在同一 Xenium-derived ground truth、同一 supervised spots 和统一 metrics 下比较，并且记录 runtime、memory、bootstrap mean/std 和 paired permutation。Figure 3 的 benchmark contract 和 reviewer controls 就是为了回应这个问题。"
  ],
  [
    "问：为什么 Rep1 no-retuning 没有直接最好？",
    "答：这个结果我们没有隐藏，而是把它作为 cross-sample domain shift 报告出来。直接迁移到 Rep1 时 JSD 为 0.3656，确实不如最佳 baseline；但 minimal retuning 只需要 25 steps 就能降到 0.1259，并超过最佳 baseline。这说明不同样本之间存在真实 domain shift，但模型可以通过很小适应预算恢复性能。"
  ],
  [
    "问：niche 的生物学意义怎么证明？",
    "答：我们不是只做聚类命名。niche 解释同时结合了细胞组成、marker enrichment、Xenium neighborhood validation、gate reliability、agent attention 和 external pathology correspondence。Figure 5 里外部 pathology agreement 达到 0.82，说明 niche 与病理分类之间有较强对应。"
  ],
  [
    "问：模型是否过平滑组织边界？",
    "答：我们做了 boundary preservation 分析。结果显示 boundary-to-interior jump ratio 为 2.36，也就是说边界区域的组成变化显著高于内部区域。此外还有 typed boundary summaries、marker validation 和 no-boundary-loss comparison，用来说明 boundary-aware 设计不是造成过平滑，而是帮助保留真实组织边缘。"
  ],
  [
    "问：这个工作最大的局限是什么？",
    "答：我会诚实地说，第一，当前主要围绕乳腺癌数据，未来需要更多癌种和组织类型验证；第二，Xenium-to-Visium ground truth 依赖空间配准和聚合半径，因此我们做了 sensitivity analysis，但仍然需要更多真实共测数据；第三，跨样本 no-retuning 会受 domain shift 影响，因此我们报告了 minimal-retuning budget，而不是把泛化说得过满。"
  ],
].forEach(([q, a]) => {
  children.push(h3(q), p(a));
});

children.push(
  h1("八、最后收尾稿"),
  p("总结一下，这项工作的贡献可以分成三句话。第一，我们构建了一个 Xenium 监督的 Visium 级乳腺癌空间反卷积 benchmark，使真实组织上的细胞组成预测可以被定量评估。第二，我们提出 WaveST-Gate，用 wavelet morphology、cell-type prototype agents 和 reliability gate 融合 H&E、空间表达和单细胞参考。第三，我们用 baseline comparison、ablation、uncertainty calibration、boundary preservation、niche interpretation、external pathology validation 和 robustness analysis，形成了一条比较完整的证据链。"),
  p("因此，我希望这项工作传达的信息不是“我们做了一个更复杂的模型”，而是“我们把空间反卷积从模型预测推进到可监督评估、可信解释和病理验证”。这也是我认为 WaveST-Gate 在投稿中最有说服力的地方。谢谢各位老师。"),
  h1("九、汇报时可以放在手边的关键词"),
  table(
    [
      ["主题", "关键词"],
      ["Benchmark", "Xenium-derived ground truth；4,992 spots；485 GT spots；115,275 cells；19 cell types；radius 55；QC and sensitivity"],
      ["Model", "H&E wavelet morphology；spot expression；scRNA prototype agents；cross-modal reliability gate；local refinement"],
      ["Main result", "JSD 0.01293；spotwise cosine 0.9933；mean Pearson 0.9283；rank 1 among 13 methods"],
      ["Reliability", "uncertainty-error Pearson 0.5263；calibration-bin Pearson 0.9619；low/intermediate/high trust states"],
      ["Biology", "boundary jump 2.36x；5 niches；15,601 external pathology spots；pathology agreement 0.82"],
      ["Robustness", "12 ablations；42 stress-test rows；split sensitivity；radius/cell-count sensitivity；Rep1 25-step adaptation"],
    ],
    [2300, tableWidth - 2300]
  )
);

const doc = new Document({
  creator: "WaveST-Gate project",
  title: "WaveST-Gate 博士汇报逐字稿",
  description: "Detailed doctoral presentation script for WaveST-Gate.",
  styles: {
    default: {
      document: { run: { font, size: 22 }, paragraph: { spacing: { line: 330 } } },
    },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font, size: 34, bold: true, color: "12324A" },
        paragraph: { spacing: { before: 320, after: 180 }, outlineLevel: 0 },
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font, size: 28, bold: true, color: "1E5A68" },
        paragraph: { spacing: { before: 260, after: 140 }, outlineLevel: 1 },
      },
      {
        id: "Heading3",
        name: "Heading 3",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font, size: 24, bold: true, color: "7A3150" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: "numbers",
        levels: [
          {
            level: 0,
            format: LevelFormat.DECIMAL,
            text: "%1.",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 520, hanging: 300 } } },
          },
        ],
      },
      {
        reference: "bullets",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "•",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 520, hanging: 300 } } },
          },
        ],
      },
    ],
  },
  sections: [
    {
      properties: { page: { size: { width: page.width, height: page.height }, margin: page.margin } },
      headers: {
        default: new Header({
          children: [
            new Paragraph({
              alignment: AlignmentType.RIGHT,
              spacing: { after: 0 },
              children: [run("WaveST-Gate 博士汇报稿", { size: 16, color: "5B6A72" })],
            }),
          ],
        }),
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              spacing: { before: 0, after: 0 },
              children: [run("Page ", { size: 16, color: "5B6A72" }), new TextRun({ children: [PageNumber.CURRENT], size: 16, font, color: "5B6A72" })],
            }),
          ],
        }),
      },
      children,
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(outPath, buffer);
  console.log(outPath);
});
