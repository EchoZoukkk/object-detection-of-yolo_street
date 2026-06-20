# Copyright 2026 Zou Kaiyu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
知识蒸馏算法模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, teacher_model, temperature=4.0, alpha=0.5, beta=0.3):
        super().__init__()
        self.teacher = teacher_model
        self.teacher.eval()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, student_pred, teacher_pred, targets, student_features, teacher_features):
        hard_loss = self._compute_hard_loss(student_pred, targets)
        soft_loss = self._compute_soft_loss(student_pred, teacher_pred)
        feat_loss = self._compute_feature_loss(student_features, teacher_features)
        total_loss = hard_loss + self.alpha * soft_loss + self.beta * feat_loss
        return {
            'total': total_loss,
            'hard': hard_loss,
            'soft': soft_loss,
            'feature': feat_loss
        }

    def _compute_hard_loss(self, pred, targets):
        return torch.tensor(0.0, requires_grad=True)

    def _compute_soft_loss(self, student_pred, teacher_pred):
        student_cls = student_pred[1] if isinstance(student_pred, tuple) else student_pred
        teacher_cls = teacher_pred[1] if isinstance(teacher_pred, tuple) else teacher_pred
        student_soft = F.log_softmax(student_cls / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_cls / self.temperature, dim=-1)
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        return kl_loss * (self.temperature ** 2)

    def _compute_feature_loss(self, student_feats, teacher_feats):
        loss = 0
        for s_feat, t_feat in zip(student_feats, teacher_feats):
            if s_feat.shape[1] != t_feat.shape[1]:
                adapt = nn.Conv2d(s_feat.shape[1], t_feat.shape[1], 1).to(s_feat.device)
                s_feat = adapt(s_feat)
            if s_feat.shape[2:] != t_feat.shape[2:]:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=False)
            loss += F.mse_loss(s_feat, t_feat.detach())
        return loss / len(student_feats)


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = []
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(module, input, output):
            self.features.append(output)

        target_layers = [6, 8, 9]
        for idx in target_layers:
            layer = list(self.model.model.children())[idx]
            layer.register_forward_hook(hook_fn)

    def forward(self, x):
        self.features = []
        out = self.model(x)
        return out, self.features