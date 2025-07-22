# 🧬 甲状腺结节病灶识别（医学图像分割）

本项目使用深度学习方法（如 Attention UNet）对甲状腺超声图像中的结节区域进行自动分割，旨在为医疗诊断提供智能辅助。

---

## 📌 项目目标

- 自动识别甲状腺图像中的结节病灶区域
- 使用迁移学习和冻结策略提升小样本性能
- 提供可复现的训练流程与评估方法

---

## 🧠 模型结构

- 使用 Attention UNet + ResNet34 + SEBlock 作为编码器
- 支持混合精度训练、early stopping
- 集成多种损失函数：DiceLoss、StabilizedBCE、Loss Transition 等
