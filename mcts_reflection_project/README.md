## 基于 MCTS 的过程反思数据生成（Qwen2-VL + MathVista）

本项目实现你在“4.3.2 基于蒙特卡洛树搜索方法的过程反思数据生成”中描述的机制，核心目标是生成用于策略学习/监督微调的**高质量反思型多模态推理数据**：

- **显式建模**多模态推理状态 \(s_t\) 与推理动作 \(a_t\)（自然语言中间推理步）
- 用 **MCTS** 系统探索多步推理路径（Selection / Expansion / Simulation / Backpropagation）
- 在 **Expansion** 阶段使用 **视觉相关性奖励**过滤候选动作（来自 Qwen2‑VL 的跨模态隐藏状态，不依赖额外对齐模型）
- 在 **Simulation** 阶段进行多次 rollout，计算：
  - 答案正确性奖励 \(R_{\text{ans}}\)
  - 反思价值 \(R_{\text{ref}}\)（由“错误路径+正确路径”生成反思型推理范式）
  - 视觉一致性累计奖励 \(R_{\text{vis}}\)
- 将三类信号融合为节点综合奖励并回溯更新，最终导出：
  - **search JSONL**：包含搜索树元信息、rollout 轨迹、错误/正确对比与反思文本
  - **train JSON**：ShareGPT 格式（`messages` + `images`），可直接喂给训练框架（如 LLaMA‑Factory）

---

## 目录结构

```
src/mcts_reflection/
  cli/                # 命令行入口
  core/               # MCTS 核心（节点/树搜索/UCT）
  datasets/           # MathVista 读取与规范化
  models/             # Qwen2-VL 推理封装（生成 + forward hidden states）
  rewards/            # 视觉相关性奖励
  reflection/         # 反思文本生成器（可用 Qwen2-VL 或外部 LLM）
  export/             # 导出 ShareGPT / 搜索日志
configs/              # 参数配置样例
```

---

## 环境安装（Python 3.12）

> 注意：不同机器/驱动的 CUDA Torch 安装方式不同，本项目**不强 pin 旧版 torch**。你可以先按以下方式安装，然后根据你的 CUDA 版本替换 torch 安装命令。

```bash
pip3 install -r requirements.txt
```

---

## 一条命令生成反思训练数据（MathVista）

```bash
python -m mcts_reflection.cli.run_mathvista_mcts \
  --data_path /path/to/mathvista.jsonl \
  --image_root /path/to/images \
  --model_path Qwen/Qwen2-VL-7B-Instruct \
  --output_search_jsonl outputs/mathvista_search.jsonl \
  --output_train_json outputs/mathvista_train_sharegpt.json \
  --max_iterations 24 \
  --num_expand 12 \
  --keep_topk_by_vision 5 \
  --num_rollouts 4 \
  --vision_layer -2 \
  --vision_tau 0.2
```

---

## 输出字段（对应你文中①-⑦）

search JSONL（每行一个样本）包含：
- **① 问题文本**：`question`
- **② 图像路径**：`image`
- **③ 高质量推理路径与最终答案**：`mcts.best.final_reasoning` / `mcts.best.pred_answer`
- **④ 标准答案**：`gt_answer`
- **⑤ 错误推理步骤**：`mcts.reflection.incorrect_step`（若生成成功）
- **⑥ 正确推理步骤**：`mcts.reflection.correct_step`（若生成成功）
- **⑦ 反思步骤**：`mcts.reflection.reflection_text`

train JSON（ShareGPT）会优先写入“反思型推理路径”，若反思对未构造成功则回退为“最好路径的完整推理”。

