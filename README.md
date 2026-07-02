# Wood Bridge Analyzer

用于分析 Rhino `.3dm` 木结构桁架桥模型的 Python 工程。V1.7 重点完成：

- 读取 Rhino 3dm 文件；
- 识别木杆构件并统计数量；
- 导出 `rod_inventory.csv`、`nodes.csv`、`members.csv`；
- 建立简化三维桁架有限元模型；
- 输出最大位移、杆件轴力、危险杆件排序；
- 生成三视图、杆件统计图、轴力图、变形图、风险图和移动荷载包络图；
- 生成 `analysis_report.md`。
- 自动读取 Rhino 文件单位，按默认 1:10 输出模型尺寸与实桥尺寸；
- 检查刚度矩阵奇异性，奇异时报告原因，并输出最小二乘近似结果用于定位风险；
- 生成节点调试图、屈曲风险表和荷载工况对比图。
- 支持人工校核覆盖支座、桥面节点、忽略杆件和强制纳入杆件；
- 输出 sanity check、保守修正结果和自动加固建议；
- 生成汇报总览图、支座荷载图、风险排序图和加固建议图。
- 输出验证检查、人工复核清单、加固优先级、展板文案和最终总览图；
- 支持修改前后两个 Rhino 模型对比，判断模型是否改得更好。
- V1.6.1 修正材料统计口径：区分模型构件数、结构有效杆件数和实际领取标准木杆数，并按排料结果计算材料分。
- V1.6.2 增加 clean_centerline_model：先导出可人工检查的中心线结构模型、节点质量报告和支座/加载节点确认图；未人工确认支座与桥面加载节点时不进入 FEM 求解。
- V1.6.3 增加中心线模型验收：检查 clean 节点/杆件连通性、支座与桥面加载节点合理性、FEM 前置可解性，并生成中心线模型评分。
- V2.0 增加可选 OpenSeesPy 求解后端：默认 `solver_backend: both` 时同时保留 numpy solver，并尝试用 OpenSeesPy 复核位移、杆力和支座反力；未安装 OpenSeesPy 时不会中断程序。

## 安装

建议使用 Python 3.11 或 3.12，并在虚拟环境中运行。`rhino3dm` 对较新的 Python 版本可能没有预编译包；如果 Python 3.14 安装失败，请换到 Python 3.12。

```bash
cd wood_bridge_analyzer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行

```bash
python3 main.py --model 桁架桥3.3dm --config config.yaml --output outputs
```

修改前后对比：

```bash
python3 main.py --model old.3dm --compare-model new.3dm --config config.yaml --output outputs
```

运行结束后查看 `outputs/`：

- `analysis_report.md`：自动分析报告；
- `model_dimensions.csv`：Rhino 单位、模型尺寸、实桥尺寸和比例检查；
- `rod_inventory.csv`：每根杆件的长度、方向、图层、是否超长和等效标准杆数量；
- `rod_inventory_summary.md`：杆件数量和材料分摘要；
- `nodes.csv`：自动聚类后的结构节点；
- `members.csv`：结构杆件网络；
- `clean_nodes.csv`：V1.6.2 清理后的中心线结构节点；
- `clean_members.csv`：V1.6.2 清理后的中心线结构杆件，每根杆件都有明确 `node_i` 和 `node_j`；
- `node_quality_report.md`：单杆节点、悬空杆件、未连接杆端和连接异常节点检查；
- `manual_node_overrides_template.csv`：人工节点修正表模板；
- `manual_node_overrides_used.csv`：本次实际读取的人工节点修正内容；
- `confidence_scores.csv`：几何识别、材料统计、节点网络、支座定义、荷载定义和 FEM 结果可信度评分；
- `centerline_validation_report.md`：V1.6.3 中心线模型验收报告；
- `centerline_model_score.csv`：中心线模型 100 分制评分；
- `connected_components.csv`：图论连通分量统计；
- `duplicate_members.csv`：重复连接同一对节点的杆件；
- `deck_node_check.md` / `deck_node_check.csv`：桥面加载节点合理性检查；
- `support_node_check.md` / `support_node_check.csv`：支座节点合理性检查；
- `fem_precheck.csv`：FEM 前置验收结果；
- `opensees_support_check.md`：OpenSeesPy 支座节点检查；
- `opensees_load_check.md`：OpenSeesPy 桥面加载节点检查；
- `opensees_analysis_log.txt`：OpenSeesPy 后端运行日志或失败原因；
- `opensees_case_summary.csv`：OpenSeesPy 各荷载工况摘要；
- `opensees_node_displacements.csv`：OpenSeesPy 节点位移；
- `opensees_member_forces.csv`：OpenSeesPy 杆件轴力；
- `opensees_reactions.csv`：OpenSeesPy 支座反力；
- `solver_comparison.md` / `solver_comparison.csv`：`solver_backend: both` 时 numpy 与 OpenSeesPy 对比；
- `close_unclustered_endpoints.csv`：距离很近但未被聚类的杆端；
- `buckling_check.csv`：受压杆欧拉屈曲检查；
- `load_case_comparison.csv`：各荷载工况最大位移、最大杆力和求解状态；
- `sanity_check.csv`：反力平衡、最大位移位置、受拉/受压区域和条件数检查；
- `validation_check.csv`：V1.7 验证检查，包含水平反力、疑似机构、异常位移/杆力等；
- `conservative_results.csv`：考虑连接滑移、施工误差、材料不确定性和屈曲安全系数后的保守结果；
- `recognition_issues.csv`：模型识别风险提示；
- `corrected_rod_inventory.csv`：应用人工修正表后的杆件清单；
- `cut_plan.csv` / `cut_plan.md`：1300mm 标准木杆排料方案；
- `material_count_comparison.md`：原始模型统计、人工统计和排料统计对比；
- `manual_member_overrides_template.csv`：人工修正表模板；
- `fix_suggestions.md`：自动生成的加固建议；
- `manual_review_checklist.md`：人工复核清单；
- `reinforcement_priority.md`：按优先级排序的结构修改建议；
- `board_text.md`：可直接放入 A1 展板的结构分析文字；
- `comparison_report.md`：传入 `--compare-model` 时生成的修改前后对比报告；
- `member_forces_*.csv`：不同荷载工况的杆件轴力、应力、利用率和屈曲风险；
- `report_summary.png`：汇报总览图；
- `top_10_risk_members.png`：最危险 10 根杆件排序；
- `fix_suggestions.png`：建议加固位置图；
- `support_and_loads.png`：支座和加载节点图；
- `node_connectivity_map.png`：节点连接数量图；
- `clean_centerline_3views.png`：清理后中心线结构模型三视图；
- `node_quality_map.png`：节点连接质量检查图；
- `support_load_manual_check.png`：支座和桥面加载节点人工确认图；
- `disconnected_members.png`：悬空或疑似断开的杆件图；
- `connectivity_map.png`：结构连通分量图；
- `deck_node_map.png`：桥面加载节点检查图；
- `support_node_map.png`：支座节点检查图；
- `centerline_review_board.png`：中心线模型验收总览图；
- `support_and_loads_opensees.png`：OpenSeesPy 支座和加载节点图；
- `opensees_force_diagram.png`：OpenSeesPy 轴力图；
- `opensees_deflection_diagram.png`：OpenSeesPy 变形图；
- `opensees_reaction_diagram.png`：OpenSeesPy 支座反力图；
- `final_review_board.png`：最终汇报总览图；
- `comparison_summary.png`：传入 `--compare-model` 时生成的对比摘要图；
- `bridge_3views.png`：平面图、立面图、剖面图；
- `node_debug.png`：节点编号和连接杆件数量；
- `rod_count_diagram.png`：杆件长度和方向统计；
- `force_diagram.png`：控制工况轴力图，红色为受拉，蓝色为受压；
- `deflection_diagram.png`：变形前后对比；
- `buckling_risk_map.png`：压杆屈曲风险图；
- `load_case_comparison.png`：不同荷载工况最大位移对比；
- `moving_load_envelope.png`：移动荷载最大位移和最大杆力曲线。

## 输入模型要求

V1 优先识别以下 Rhino 对象：

- `LineCurve` 或可读取起终点的曲线；
- `Extrusion`、`Brep` 等有包围盒的实体对象。

建议把木杆放在名称包含以下关键词的图层：

```text
wood, rod, member, timber, 木, 杆
```

如果你的模型图层命名不同，请修改 `config.yaml` 中的：

```yaml
model:
  rod_layer_keywords: [...]
```

如果希望扫描所有图层，可设为：

```yaml
model:
  rod_layer_keywords: []
```

## 单位和比例

任务模型为 1:10 小模型，默认：

```yaml
model:
  scale: 10.0
```

程序会把模型尺寸换算为真实尺寸。模型中的 `3mm x 8mm x 130mm` 对应真实 `30mm x 80mm x 1300mm`。

节点聚类容差可在 `config.yaml` 修改：

```yaml
model:
  node_cluster_tolerance_model_mm: 5.0
```

## V1.6.2 模型层级

报告会明确区分四个模型层级：

- 展示模型：Rhino 3dm 中的实体、曲线、节点块等原始对象；
- 施工材料统计模型：经过人工杆件修正与 1300mm 标准木杆排料后的材料模型；
- 结构中心线计算模型：每根有效木杆只保留一条中心线，输出 `clean_nodes.csv` 和 `clean_members.csv`；
- FEM 求解模型：只有在支座节点和桥面加载节点人工确认后，才进入简化三维桁架求解。

## V1.6.3 中心线模型验收

V1.6.3 会先读取 `clean_nodes.csv` 和 `clean_members.csv`，用图论方法检查结构网络。只有同时满足以下条件，程序才进入可靠 FEM 分析：

- clean 中心线结构是单一连通体；
- 不存在大量单杆节点、悬空杆件、未连接杆端或重复杆件；
- `fixed_nodes`、`roller_nodes` 和 `deck_nodes` 已人工确认；
- 支座位于两端低点，约束足以限制刚体位移；
- 桥面加载节点位于桥面高度范围，并沿跨度合理分布；
- FEM 前置刚度矩阵检查不奇异，条件数未异常偏大。

如果验收不通过，报告会写明：

```text
支座节点和桥面加载节点尚未人工确认，当前模型不能进入可靠受力分析。
```

并在 `centerline_validation_report.md` 和 `analysis_report.md` 中列出阻断原因。

## V2.0 OpenSeesPy 后端

OpenSeesPy 是可选后端，不会替代现有 numpy FEM solver。配置项：

```yaml
solver_backend: both  # 可选 numpy, openseespy, both

opensees:
  element_type: truss
  ndm: 3
  ndf: 3
  use_corotational_truss: false
  material:
    E_MPa: 10000.0
    area_mm2: 2400.0
    density_kg_per_m3: 500.0
  analysis:
    load_cases:
      - self_weight
      - midspan_person
      - group_uniform
      - eccentric_walk
    displacement_limit_mm: 500.0
```

如果没有安装 OpenSeesPy，程序会继续完成材料统计、中心线验收和 numpy 结果，并在日志中写入：

```text
OpenSeesPy backend unavailable. Install with: pip install openseespy
```

OpenSeesPy 后端只读取 `clean_nodes.csv` 和 `clean_members.csv`，不会直接读取 Rhino 实体模型。进入 OpenSeesPy 分析前必须人工指定 `fixed_nodes`、`roller_nodes` 和 `deck_nodes`。

人工节点修正表 `manual_node_overrides.csv` 支持：

```csv
action,node_ids,target_node_id,member_id,end,node_id,note
merge_nodes,1;2,1,,,,把两个近似节点合并为一个真实节点
force_connect,,,3,start,1,把 member 3 起点强制连接到 node 1
no_connect,,,,,,记录交叉但不连接的位置
```

## 有限元模型假设

程序使用简化三维桁架模型：

- 每个节点有 `Ux, Uy, Uz` 三个平移自由度；
- 每根杆件只承受轴力；
- 节点按铰接处理；
- 杆件刚度为 `AE/L`；
- 默认弹性模量 `E = 10000 MPa`，可在 `config.yaml` 修改；
- 对受压杆做欧拉屈曲检查，截面惯性矩采用 `30mm x 80mm` 矩形弱轴。

重要：该分析不能替代实体加载实验。节点刚度、木材缺陷、连接滑移、绳索预拉力和施工误差都需要通过真实测试修正。

## 支座设置

V1.6.2 开始，默认不再用自动推断支座和桥面节点直接进入 FEM。若 `fixed_nodes`、`roller_nodes`、`deck_nodes` 没有人工指定，程序只导出中心线模型、节点质量报告和提示图，并在报告中写明“请先人工确认支座和桥面加载节点”。

旧版自动推断逻辑仍保留在代码中，但只建议作为人工选择节点时的参考：

- 一端固定铰支座：限制 `Ux, Uy, Uz`；
- 另一端滚动支座：限制 `Ux, Uz`。

也可以在 `config.yaml` 中手动指定：

```yaml
supports:
  auto_detect: false
  fixed_nodes: [0, 14]
  roller_nodes: [12, 26]
```

桥面加载节点也可以手动筛选：

```yaml
bridge:
  deck_node_filter:
    node_ids: null
    x_range_mm: null
    y_range_mm: [-650, 650]
    z_range_mm: [0, 200]
```

## 人工校核模式

优先使用 `manual_overrides`。没有填写时才使用自动识别。

```yaml
manual_overrides:
  support_nodes:
    fixed: [0, 14]
    roller: [12, 26]
  deck_nodes: [0, 1, 2, 3]
  ignored_members: [5, 8]
  force_include_members: []
```

## 材料统计修正

V1.6.1 之后，材料分使用 `stock_wood_count`，不是模型对象数量。

- `model_member_count`：3dm 中识别到的杆状构件数量。
- `structural_member_count`：排除/合并/修正后用于统计的有效构件数量。
- `stock_wood_count`：按 1300mm 标准木杆排料后实际需要领取的木杆数量。

可在 `manual_member_overrides.csv` 中人工修正：

```csv
member_id,action,corrected_length_mm,group_id,note
16,duplicate_of,,54,duplicate member
21,shorten,1280,,trimmed effective length
30,non_structural,,,helper line
```

保守修正系数：

```yaml
conservative_factors:
  connection_slip_factor: 1.2
  construction_error_factor: 1.15
  material_uncertainty_factor: 1.15
  buckling_safety_factor: 1.5
```

## V2.1 标准木杆近似排料

V2.1 新增标准木杆近似长度排料统计，只影响材料统计和材料成本分，不会修改 `clean_members.csv` 的真实杆长，也不会影响 numpy FEM 或 OpenSeesPy 的结构计算。

程序优先读取 `corrected_rod_inventory.csv`；如果该文件不存在，则读取 `clean_members.csv`。只统计 `member_type = wood` 的有效木杆，排除绳索、金属节点、支座标记、加载标记、辅助线和非结构对象。

默认设置为：

```yaml
material_stock_counting:
  enabled: true
  stock:
    stock_length_mm: 1300
  length_rounding:
    enabled: true
    method: nearest
    step_mm: 50
  pairing:
    enabled: true
    target_length_mm: 1300
    pair_tolerance_mm: 25
    max_pieces_per_stock: 2
  manual_compare:
    manual_stock_count: 46
    prefer_manual_stock_count: true
```

运行示例：

```bash
python main.py --model 桁架桥3.3dm --config config.yaml --output outputs --enable-stock-pairing --round-step 50 --pair-tolerance 25 --manual-stock-count 46
```

新增输出：

- `rounded_member_lengths.csv`：每根有效木杆的真实长度、近似长度和取整误差；
- `paired_stock_cut_plan.csv`：每根 1300mm 标准木杆的裁切组合；
- `paired_stock_cut_plan.md`：中文裁切方案；
- `material_stock_summary.md` / `material_stock_summary.json`：最终标准木杆数量、人工统计对比和材料成本分；
- `oversized_members.csv`：超过 1300mm、需要拆分或重新设计的构件；
- `length_rounding_distribution.png`、`stock_pairing_plan.png`、`material_stock_count_summary.png`。

当程序统计和人工统计差异较大时，应优先复核是否有非结构杆件被计入、重复杆件未排除、超长杆件未拆分，或是否需要允许每根标准木杆裁切三段以上。

注意：`model_member_count` 是模型中的木杆构件段数，`stock_wood_count` 是最终用于材料成本分的 1300mm 标准木杆领取数量。两者不是同一个统计口径。有人工复核数量时，默认用 `manual_stock_count` 作为最终 `stock_wood_count`，程序排料结果会保存在 `program_stock_wood_count` 中用于核对。

## TODO

- 更精确地从 Rhino Brep/Extrusion 中提取任意方向杆件中心线；
- 区分木杆、金属节点、绳索和桥面铺板；
- 自动识别桥面平面 X 形拉结与顶部横向联系；
- 支持绳索只受拉单元；
- 加入连接节点滑移和半刚性节点模型；
- 输出交互式 3D HTML 查看器。
